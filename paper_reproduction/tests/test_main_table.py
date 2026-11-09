"""Tests of selective Main Table aggregation (CPU only)."""
import json
import tempfile
import unittest
from pathlib import Path
from main_table import TASKS, MODELS, read_report, render
from run_main_table import DEFAULTS, ALIASES, parse_list, select_data

class TestMainTable(unittest.TestCase):
    def sample(self, tasks=None):
        tasks = tasks or [name for _, name in TASKS]
        return dict(n_examples=200 * len(tasks),n_missing=0,
                    overall=dict(gt_acc=29.,ala=40.),
                    per_task={name:dict(n=200,gt_acc=29.,ala=40.) for name in tasks})

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

    def test_subset_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"qwen").mkdir()
            tasks=["gender_prediction", "emotion_recognition"]
            for kind in ("baseline","iaan"):
                (root/"qwen"/f"{kind}_report.json").write_text(
                    json.dumps(self.sample(tasks)))
            result=render(root, models=["qwen"], tasks=tasks)
            self.assertIn("Gender",result)
            self.assertIn("Emotion",result)
            self.assertNotIn("Intonation",result)

    def test_rejects_missing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.json"
            obj=self.sample()
            obj["n_examples"]=700
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                read_report(path)

    def test_paper_defaults(self):
        self.assertEqual(DEFAULTS, {"af3":(50,16.), "qwen":(50,12.), "kimi":(200,12.)})
        self.assertEqual(parse_list("qwen,kimi",set(DEFAULTS)),["qwen","kimi"])
        self.assertEqual(parse_list("emotion,gender",set(x for _,x in TASKS),ALIASES),
                         ["emotion_recognition","gender_prediction"])

    def test_filter_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"all.json"
            samples=[{"id":"gender_prediction__1"},{"id":"emotion_recognition__2"}]
            source.write_text(json.dumps(samples))
            out=root/"subset.json"
            self.assertEqual(select_data(source,out,["gender_prediction"]),1)
            self.assertEqual(json.loads(out.read_text())[0]["id"], samples[0]["id"])\n            self.assertEqual(json.loads(out.read_text())[0]["_iaan_index"], 0)

if __name__=="__main__":
    unittest.main()
