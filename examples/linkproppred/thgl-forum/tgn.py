import numpy as np
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
import timeit


import math
import time
import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator, MeanAggregator, SumAggregator, MaxAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

AGGREGATOR_REGISTRY = {
    "last": LastAggregator,
    "mean": MeanAggregator,
    "sum": SumAggregator,
    "max": MaxAggregator,
}

wandb_module = None
wandb_run = None
wandb_step_counter = 0


def get_aggregator_cls(name: str):
    key = name.lower()
    if key not in AGGREGATOR_REGISTRY:
        raise ValueError(f"Unsupported aggregator '{name}'. Choices: {list(AGGREGATOR_REGISTRY.keys())}")
    return AGGREGATOR_REGISTRY[key], key


def init_wandb(args, run_name: str, config: dict):
    global wandb_module, wandb_run, wandb_step_counter
    if not args.wandb:
        return
    try:
        import wandb as wandb_lib
    except ImportError:
        print("WARNING: wandb is not installed; disabling wandb logging.")
        args.wandb = False
        return

    wandb_module = wandb_lib
    wandb_run = wandb_module.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        config=config,
    )
    wandb_step_counter = 0
    print(f"INFO: Initialized Weights & Biases run: {wandb_run.name}")


def log_to_wandb(payload: dict):
    global wandb_step_counter
    if wandb_module is None or wandb_run is None:
        return
    wandb_module.log(payload, step=wandb_step_counter)
    wandb_step_counter += 1


def _truncate_split(split_data, fraction, split_name):
    if split_data is None or fraction >= 1.0:
        return split_data
    total_events = split_data.msg.size(0)
    keep_events = max(1, int(total_events * fraction))
    if keep_events >= total_events:
        return split_data
    print(f"INFO: {split_name} split truncated to {keep_events}/{total_events} events (fraction={fraction:.6f})")
    return split_data[:keep_events]

# ==========
# ========== Define helper function...
# ==========

def train(epoch_idx: int):
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch_idx, batch in enumerate(train_loader, start=1):
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg, rel = batch.src, batch.dst, batch.t, batch.msg, batch.edge_type

        num_pos = src.size(0)
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (num_pos * TRAIN_NEG_SAMPLES,),
            dtype=torch.long,
            device=device,
        )
        neg_dst = neg_dst.view(TRAIN_NEG_SAMPLES, num_pos)

        expanded_src = src.unsqueeze(0).expand(TRAIN_NEG_SAMPLES, -1)
        neg_src = expanded_src.reshape(-1)
        flat_neg_dst = neg_dst.reshape(-1)

        n_id = torch.cat([src, pos_dst, flat_neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[neg_src]], z[assoc[flat_neg_dst]])
        neg_out = neg_out.view(TRAIN_NEG_SAMPLES, num_pos, -1)
        neg_out = neg_out.mean(dim=0)

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        batch_loss = float(loss.detach())
        total_loss += batch_loss * batch.num_events

        if LOG_EVERY > 0 and batch_idx % LOG_EVERY == 0:
            est_batches = max(1, NUM_TRAIN_BATCHES)
            global_step = (epoch_idx - 1) * est_batches + batch_idx
            print(
                f"[Train] Epoch {epoch_idx} Batch {batch_idx}/{est_batches} "
                f"loss={batch_loss:.4f} pos={float(pos_out.mean()):.4f} neg={float(neg_out.mean()):.4f}"
            )
            log_to_wandb(
                {
                    "train_batch_loss": batch_loss,
                    "train_batch": batch_idx,
                    "epoch": epoch_idx,
                    "run_batch_step": global_step,
                }
            )

    return total_loss / max(1, train_data.num_events)


@torch.no_grad()
def test(loader, neg_sampler, split_mode, total_batches=None):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    for batch_idx, pos_batch in enumerate(loader, start=1):
        pos_src, pos_dst, pos_t, pos_msg, pos_rel = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
            pos_batch.edge_type
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, pos_rel, split_mode=split_mode)

        # pos_msg_new = torch.cat([pos_msg,pos_rel.unsqueeze(dim=1)], dim=1)   


        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            pos_scores = y_pred[0, :].squeeze(dim=-1).detach().cpu().numpy()
            neg_scores = y_pred[1:, :].squeeze(dim=-1).detach().cpu().numpy()
            input_dict = {
                "y_pred_pos": pos_scores.reshape(1, -1),
                "y_pred_neg": neg_scores,
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        if LOG_EVERY > 0 and total_batches is not None and batch_idx % LOG_EVERY == 0:
            print(f"[Eval-{split_mode}] Processed {batch_idx}/{total_batches} batches")

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics

# ==========
# ==========
# ==========


# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

DEFAULT_DATA = "thgl-forum"

# ========== set parameters...
args, _ = get_args()
if not args.data or args.data == "tgbl-wiki":
    args.data = DEFAULT_DATA
DATA = args.data
print("INFO: Arguments:", args)
aggregator_cls, aggregator_name = get_aggregator_cls(args.aggr)
EDGE_EMB_DIM = args.edge_emb_dim
if args.log_every < 0:
    raise ValueError("log_every must be non-negative")
LOG_EVERY = args.log_every
NUM_WORKERS = max(0, args.num_workers)
print(f"INFO: Using aggregator: {aggregator_name}, Edge emb dim: {EDGE_EMB_DIM}, log_every={LOG_EVERY}, num_workers={NUM_WORKERS}")

LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
USE_EDGE_TYPE = True
USE_NODE_TYPE = True
TRAIN_NEG_SAMPLES = max(1, args.num_neg_samples)
CHECKPOINT_EVERY = max(0, args.checkpoint_every)



MODEL_NAME = 'TGN'
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

print ("there are {} nodes and {} edges".format(dataset.num_nodes, dataset.num_edges))
print ("there are {} relation types".format(dataset.num_rels))


timestamp = data.t
head = data.src
tail = data.dst
edge_type = data.edge_type #relation
edge_type_dim = len(torch.unique(edge_type))

embed_edge_type = torch.nn.Embedding(edge_type_dim, EDGE_EMB_DIM).to(device)
with torch.no_grad():
    edge_type_embeddings = embed_edge_type(edge_type)


if USE_EDGE_TYPE:
    data.msg = torch.cat([data.msg, edge_type_embeddings], dim=1)

#! node type is a property of the dataset not the temporal data as temporal data has one entry per edge
node_type = dataset.node_type #node type
neg_sampler = dataset.negative_sampler

data.__setattr__("node_type", node_type)

print ("shape of edge type is", edge_type.shape)
print ("shape of node type is", node_type.shape)

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

split_fraction = max(0.0, min(1.0, args.split_frac))
if split_fraction <= 0:
    raise ValueError("split_frac must be greater than 0")

train_data = _truncate_split(train_data, split_fraction, "train")
val_data = _truncate_split(val_data, split_fraction, "val")
test_data = _truncate_split(test_data, split_fraction, "test")
print ("finished loading PyG data")

train_events = int(train_data.num_events)
val_events = int(val_data.num_events)
test_events = int(test_data.num_events)

wandb_config = {
    "dataset": DATA,
    "model": MODEL_NAME,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "mem_dim": MEM_DIM,
    "time_dim": TIME_DIM,
    "emb_dim": EMB_DIM,
    "edge_emb_dim": EDGE_EMB_DIM,
    "num_neighbors": NUM_NEIGHBORS,
    "patience": PATIENCE,
    "tolerance": TOLERANCE,
    "num_epoch": NUM_EPOCH,
    "num_run": NUM_RUNS,
    "aggr": aggregator_name,
    "split_frac": split_fraction,
    "num_workers": NUM_WORKERS,
    "train_events": train_events,
    "val_events": val_events,
    "test_events": test_events,
    "num_nodes": dataset.num_nodes,
    "num_edges": dataset.num_edges,
    "num_neighbors": NUM_NEIGHBORS,
    "gnn_layers": 1,
}
if args.wandb_run_name:
    run_name = args.wandb_run_name
else:
    sanitized_lr = f"{LR:.1e}".replace("+", "").replace("-", "m").replace(".", "p")
    run_name = (
        f"{MODEL_NAME}_{DATA}_aggr-{aggregator_name}"
        f"_bs{BATCH_SIZE}_lr{sanitized_lr}_mem{MEM_DIM}_time{TIME_DIM}"
        f"_emb{EMB_DIM}_neigh{NUM_NEIGHBORS}_layers1_epochs{NUM_EPOCH}"
    )
    run_name = run_name.replace("__", "_")
init_wandb(args, run_name, wandb_config)

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

def _safe_len(loader, default_events):
    try:
        return len(loader)
    except TypeError:
        return max(1, math.ceil(default_events / max(1, BATCH_SIZE)))

NUM_TRAIN_BATCHES = _safe_len(train_loader, train_data.num_events)
NUM_VAL_BATCHES = _safe_len(val_loader, max(1, val_data.num_events))
NUM_TEST_BATCHES = _safe_len(test_loader, max(1, test_data.num_events))

start_time = timeit.default_timer()

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())


print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

log_dir = Path(results_path) / "training_logs"
log_dir.mkdir(parents=True, exist_ok=True)
metrics_log_path = log_dir / f"{run_name}_metrics.log"
if not metrics_log_path.exists():
    with metrics_log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Run: {run_name}\n")
        fh.write(f"# Dataset: {DATA}\n")
        fh.write(f"# Aggregator: {aggregator_name}\n")
        fh.write("# timestamp | section | epoch | metric | value\n")

checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (Path(results_path) / "checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

def append_metric_log(section: str, epoch_value: float, metric_name: str, metric_value: float) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with metrics_log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{timestamp} | {section} | {epoch_value:.2f} | {metric_name} | {float(metric_value):.6f}\n"
        )

def save_periodic_checkpoint(epoch_value: int, run_index: int, model_dict: dict, optimizer_obj) -> None:
    if CHECKPOINT_EVERY <= 0:
        return
    if epoch_value % CHECKPOINT_EVERY != 0:
        return
    checkpoint_payload = {
        "epoch": epoch_value,
        "run_idx": run_index,
        "model_state": {name: module.state_dict() for name, module in model_dict.items()},
        "optimizer_state": optimizer_obj.state_dict(),
        "args": vars(args),
    }
    ckpt_path = checkpoint_dir / f"{run_name}_run{run_index}_epoch{epoch_value:03d}.pth"
    torch.save(checkpoint_payload, ckpt_path)
    print(f"INFO: Saved periodic checkpoint to {ckpt_path}")

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # neighhorhood sampler
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

    aggregator_module = aggregator_cls()

    # define the model end-to-end
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        MEM_DIM,
        TIME_DIM,
        message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
        aggregator_module=aggregator_module,
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM,
        out_channels=EMB_DIM,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

    model = {'memory': memory,
            'gnn': gnn,
            'link_pred': link_pred}

    optimizer = torch.optim.Adam(
        set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
        lr=LR,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples
    dataset.load_val_ns()

    val_perf_list = []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss = train(epoch)
        epoch_step = epoch + run_idx * NUM_EPOCH
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )
        log_to_wandb({"epoch": epoch, "train_loss": loss, "run_idx": run_idx})
        append_metric_log("train", epoch, "loss", loss)

        # validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_loader, neg_sampler, split_mode="val", total_batches=NUM_VAL_BATCHES)
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_payload = {
            f"val_{metric}": perf_metric_val,
            "epoch": epoch,
            "run_idx": run_idx,
        }
        if metric == "mrr":
            val_payload["val/mrr"] = perf_metric_val
        log_to_wandb(val_payload)
        append_metric_log("val", epoch, metric, perf_metric_val)
        val_perf_list.append(perf_metric_val)

        save_periodic_checkpoint(epoch, run_idx, model, optimizer)

        # check for early stopping
        if early_stopper.step_check(perf_metric_val, model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")
    log_to_wandb({"train_val_time": train_val_time, "run_idx": run_idx})

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

    # loading the test negative samples
    dataset.load_test_ns()

    # final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(test_loader, neg_sampler, split_mode="test", total_batches=NUM_TEST_BATCHES)

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
    test_payload = {
        f"test_{metric}": perf_metric_test,
        "test_time": test_time,
        "run_idx": run_idx,
    }
    if metric == "mrr":
        test_payload["test/mrr"] = perf_metric_test
    log_to_wandb(test_payload)
    append_metric_log("test", epoch, metric, perf_metric_test)

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  f'val {metric}': val_perf_list,
                  f'test {metric}': perf_metric_test,
                  'test_time': test_time,
                  'tot_train_val_time': train_val_time
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

if wandb_module is not None and wandb_run is not None:
    wandb_module.finish()

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")

# #* load numpy arrays instead
# from tgb.linkproppred.dataset import LinkPropPredDataset

# # data loading
# dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
# data = dataset.full_data  
# metric = dataset.eval_metric
# sources = dataset.full_data['sources']
# print ("finished loading numpy arrays")
