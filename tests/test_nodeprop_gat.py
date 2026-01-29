import subprocess
import sys
from pathlib import Path
import unittest


class TestNodePropGAT(unittest.TestCase):
    def test_nodeprop_gat_runs_and_reports_metrics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        exports_root = repo_root / "relbench_exports" / "tgbn-genre"
        if not exports_root.exists():
            self.skipTest("relbench_exports/tgbn-genre not present; run export scripts first.")

        cmd = [
            str(repo_root / ".venv" / "bin" / "python"),
            str(repo_root / "baselines" / "graphsage_nodeprop.py"),
            "--dataset",
            "tgbn-genre",
            "--model",
            "gat",
            "--epochs",
            "2",
            "--batch_size",
            "1024",
            "--fanouts",
            "10,5",
            "--emb_dim",
            "64",
            "--hidden_dim",
            "64",
            "--num_heads",
            "4",
            "--num_neg_eval",
            "50",
            "--max_train_events",
            "10000",
            "--max_eval_events",
            "2000",
            "--adj",
            "val",
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        out = proc.stdout
        self.assertIn("val_mrr=", out)
        self.assertIn("test_mrr=", out)


if __name__ == "__main__":
    unittest.main()

