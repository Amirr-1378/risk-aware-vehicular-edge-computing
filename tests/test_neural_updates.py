"""Hand-computed targets, QR gradients, masking, and risk signs (optional torch)."""

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from risk_aware_v2x.agents.updates import (
        bootstrap_cost_targets, cost_scores, masked_cost_actions, quantile_huber_loss,
    )
    from risk_aware_v2x.risk import mean_cvar_cost


@unittest.skipUnless(HAS_TORCH, "install the optional learning dependencies to test neural updates")
class NeuralUpdateTests(unittest.TestCase):
    def test_dqn_and_double_dqn_select_and_evaluate_different_nets(self):
        target = torch.tensor([[[2.], [8.], [-1e6]]], requires_grad=True)
        online = torch.tensor([[[9.], [1.], [-1e6]]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        costs, discounts = torch.tensor([3.]), torch.tensor([.81])
        dqn = bootstrap_cost_targets(costs, discounts, target, mask)
        double = bootstrap_cost_targets(costs, discounts, target, mask, selector_atoms=online)
        self.assertAlmostEqual(float(dqn[0, 0]), 4.62, places=5)
        self.assertAlmostEqual(float(double[0, 0]), 9.48, places=5)
        self.assertFalse(dqn.requires_grad)
        self.assertFalse(double.requires_grad)

    def test_terminal_and_zero_discount_skip_nan_next_atoms_and_empty_masks(self):
        target = torch.tensor([[[float("nan")], [float("inf")]], [[4.], [1.]]])
        masks = torch.tensor([[False, False], [True, True]])
        result = bootstrap_cost_targets(torch.tensor([7., 3.]), torch.tensor([0., .5]), target, masks)
        torch.testing.assert_close(result, torch.tensor([[7.], [3.5]]))

    def test_cvar_selector_changes_entire_target_distribution_not_only_mean(self):
        atoms = torch.tensor([[[0.] * 9 + [100.], [12.] * 10, [-1e6] * 10]])
        masks = torch.tensor([[True, True, False]])
        neutral = bootstrap_cost_targets(torch.tensor([2.]), torch.tensor([.5]), atoms, masks)
        risky = bootstrap_cost_targets(torch.tensor([2.]), torch.tensor([.5]), atoms, masks, risk_weight=.5)
        torch.testing.assert_close(neutral, 2. + .5 * atoms[:, 0])
        torch.testing.assert_close(risky, torch.full((1, 10), 8.))

    def test_torch_risk_scores_match_independent_scalar_fractional_tail(self):
        atoms = torch.tensor([[[float(value) for value in range(15)], [-3.] * 15]], dtype=torch.float64)
        for alpha in (0., .5, .9):
            for weight in (0., .5, 1.):
                scores = cost_scores(atoms, alpha=alpha, risk_weight=weight)
                for index in range(2):
                    expected = mean_cvar_cost(atoms[0, index].tolist(), alpha=alpha, risk_weight=weight)
                    self.assertAlmostEqual(float(scores[0, index]), expected, places=12)

    def test_masked_nan_and_lowest_index_ties(self):
        atoms = torch.tensor([[[4., 3.], [3., 4.], [float("nan"), -1e6]]])
        self.assertEqual(int(masked_cost_actions(atoms, torch.tensor([[True, True, False]]))[0]), 0)
        with self.assertRaises(ValueError):
            masked_cost_actions(atoms, torch.tensor([[True, True, True]]))
        with self.assertRaises(ValueError):
            masked_cost_actions(atoms, torch.tensor([[False, False, False]]))

    def test_pairwise_quantile_loss_and_gradient_by_hand(self):
        predictions = torch.tensor([[0., 2.]], requires_grad=True)
        targets = torch.tensor([[1., 3.]], requires_grad=True)
        loss = quantile_huber_loss(predictions, targets)
        self.assertAlmostEqual(float(loss.detach()), .625)
        loss.backward()
        torch.testing.assert_close(predictions.grad, torch.tensor([[-.25, -.25]]))
        self.assertIsNone(targets.grad)

    def test_huber_threshold_scaling_and_batch_reduction(self):
        prediction = torch.tensor([[0.]])
        self.assertAlmostEqual(float(quantile_huber_loss(prediction, torch.tensor([[3.]]))), 1.25)
        self.assertAlmostEqual(float(quantile_huber_loss(prediction, torch.tensor([[.5]]), kappa=2.)), .0625)
        one = quantile_huber_loss(prediction, torch.tensor([[3.]]))
        two = quantile_huber_loss(prediction.repeat(2, 1), torch.tensor([[3.], [3.]]))
        torch.testing.assert_close(one, two)

    def test_quantile_fraction_order_is_not_sorted_during_loss(self):
        ordered = quantile_huber_loss(torch.tensor([[0., 2.]]), torch.tensor([[1., 3.]]))
        crossed = quantile_huber_loss(torch.tensor([[2., 0.]]), torch.tensor([[1., 3.]]))
        self.assertNotEqual(float(ordered), float(crossed))

    def test_invalid_confidence_loss_inputs_and_discounts(self):
        with self.assertRaises(ValueError):
            cost_scores(torch.ones(1, 1, 2), alpha=1.)
        with self.assertRaises(ValueError):
            quantile_huber_loss(torch.tensor([[float("nan")]]), torch.ones(1, 1))
        with self.assertRaises(ValueError):
            quantile_huber_loss(torch.ones(1, 1), torch.ones(1, 1), kappa=0.)
        with self.assertRaises(ValueError):
            bootstrap_cost_targets(torch.tensor([1.]), torch.tensor([1.1]), torch.ones(1, 1, 1),
                                   torch.tensor([[True]]))
