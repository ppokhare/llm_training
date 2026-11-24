"""Invalid backward passes must be distinguished before an optimizer update."""
import unittest
import torch
from scripts.train_grpo import GradientFailure, check_adapter_gradients


class GradientChecks(unittest.TestCase):
    def test_finite_and_zero_gradients_are_valid(self):
        model = torch.nn.Linear(2, 1)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        self.assertEqual(check_adapter_gradients(model)['gradient_tensors'], 2)

    def test_missing_gradients_are_distinguished(self):
        model = torch.nn.Linear(2, 1)
        with self.assertRaisesRegex(GradientFailure, '"gradient_tensors": 0'):
            check_adapter_gradients(model)

    def test_nonfinite_gradients_identify_parameter(self):
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.full_like(model.weight, float('nan'))
        model.bias.grad = torch.zeros_like(model.bias)
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(GradientFailure, '"nonfinite_gradient_names": \\["weight"\\]'):
            check_adapter_gradients(model)
        self.assertTrue(torch.equal(before, model.weight))


if __name__ == '__main__':
    unittest.main()
