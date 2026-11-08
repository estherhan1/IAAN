"""Validation tests for full-main-table aggregation; no GPU needed."""
import json
import tempfile
import unittest
from pathlib import Path
from main_table import TASKS, MODELS, read_report, render

class TestMainTable(unittest.TestCase):
    def sample(self):
        return dict(n_examples=2000,n_missing=0,overall=dict(gt_acc=29.,ala=40.),
                    per_task={name:dict(n=200,gt_acc=29.,ala=40.) for _,name in TASKS})

    def test_complete_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in MODELS:
                (root/model).mkdir()
                for kind in ("baseline","iaan"):
                    (root/model/f"{kind}_report.json").write_text(json.dumps(self.sample()))
            table=render(root)
            self.assertEqual(table.count("| qwen |"),2)
            self.assertIn("| kimi | iaan |",table)

    def test_rejects_missing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.json"
            obj=self.sample()
            obj["n_examples"]=700
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                read_report(path)

if __name__ == "__main__":
    unittest.main()
