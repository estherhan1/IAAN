"""Unit tests for the model-independent core (no model downloads required)."""
import unittest
import torch
from torch import nn

from iaan_core import gaussian_reference, collect_means, select_topk, amplify_fc2_inputs


class TestIAAN(unittest.TestCase):
    def test_reference_is_deterministic_and_moment_matched(self):
        x = torch.arange(100, dtype=torch.float32)
        a = gaussian_reference(x, 7)
        b = gaussian_reference(x, 7)
        self.assertTrue(torch.equal(a, b))
        self.assertAlmostEqual(a.mean().item(), x.mean().item(), places=4)
        self.assertAlmostEqual(a.std(unbiased=False).item(),
                               x.std(unbiased=False).item(), places=4)

    def test_topk_is_signed_and_global(self):
        selected = select_topk(
            [torch.tensor([0., 10.]), torch.tensor([3., 2.])],
            [torch.tensor([99., 0.]), torch.tensor([0., 0.])], 2)
        self.assertEqual(set(selected), {0, 1})
        self.assertEqual(selected[0].tolist(), [1])
        self.assertEqual(selected[1].tolist(), [0])

    def test_hooks_capture_inputs_and_restore(self):
        fc2 = nn.Linear(3, 2, bias=False)
        x = torch.tensor([[[1., 2., 3.], [3., 4., 5.]]])
        means = collect_means(lambda: fc2(x), [fc2])
        self.assertTrue(torch.allclose(means[0], torch.tensor([2., 3., 4.])))
        before = fc2(x).detach()
        with amplify_fc2_inputs([fc2], {0: torch.tensor([1])}, 3):
            expected = x.clone()
            expected[..., 1] *= 3
            self.assertTrue(torch.allclose(fc2(x), torch.nn.functional.linear(expected, fc2.weight, fc2.bias)))
        self.assertTrue(torch.allclose(before, fc2(x)))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            select_topk([torch.zeros(2)], [torch.zeros(2)], 3)
        with self.assertRaises(ValueError):
            with amplify_fc2_inputs([], {}, 1):
                pass


if __name__ == "__main__":
    unittest.main()
