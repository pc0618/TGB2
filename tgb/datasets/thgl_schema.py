import torch
from torch_geometric.data import TemporalData

AGGREGATED_RELATION_IDS = {
    "user_issue_events": 0,
    "issue_repo_events": 1,
    "user_pr_events": 2,
    "pr_repo_events": 3,
    "user_repo_events": 4,
    "repo_repo_events": 5,
}

EVENT_NAME_TO_ID = {
    "opened": 0,
    "closed": 1,
    "reopened": 2,
    "added_collaborator": 3,
    "forked_from": 4,
}

EDGE_ID_TO_SCHEMA = {
    0: ("user_issue_events", "closed"),          # U_SE_C_I
    1: ("issue_repo_events", "closed"),          # I_AO_C_R
    2: ("issue_repo_events", "opened"),          # I_AO_O_R
    3: ("user_pr_events", "opened"),             # U_SO_O_P
    4: ("pr_repo_events", "opened"),             # P_AO_O_R
    5: ("user_issue_events", "opened"),          # U_SE_O_I
    6: ("user_pr_events", "closed"),             # U_SO_C_P
    7: ("pr_repo_events", "closed"),             # P_AO_C_R
    8: ("user_pr_events", "reopened"),           # U_SO_R_P
    9: ("pr_repo_events", "reopened"),           # P_AO_R_R
    10: ("user_issue_events", "reopened"),       # U_SE_RO_I
    11: ("issue_repo_events", "reopened"),       # I_AO_RO_R
    12: ("user_repo_events", "added_collaborator"),  # U_CO_A_R
    13: ("repo_repo_events", "forked_from"),         # R_FO_R
}

SCHEMA_VARIANTS = ("default18", "agg10")


def convert_temporal_data_variant(data: TemporalData, variant: str) -> TemporalData:
    if variant not in SCHEMA_VARIANTS:
        raise ValueError(f"Unsupported schema variant '{variant}'")
    if variant == "default18":
        return data
    return _convert_to_agg10(data)


def _convert_to_agg10(data: TemporalData) -> TemporalData:
    converted = data.clone()
    edge_type = converted.edge_type.long()
    new_edge_type = torch.empty_like(edge_type)
    event_feats = torch.empty((edge_type.numel(), 1), dtype=torch.float32)

    for original_id, (agg_name, event_name) in EDGE_ID_TO_SCHEMA.items():
        mask = edge_type == original_id
        if not torch.any(mask):
            continue
        new_edge_type[mask] = AGGREGATED_RELATION_IDS[agg_name]
        event_feats[mask, 0] = EVENT_NAME_TO_ID[event_name]

    converted.edge_type = new_edge_type
    msg = converted.msg.float()
    event_feats = event_feats.to(msg.device)
    converted.msg = torch.cat([msg, event_feats], dim=1)
    return converted
