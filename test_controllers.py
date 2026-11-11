"""Regression checks for controller boundaries, inference, and training."""

import unittest
from unittest.mock import patch

import controllers as lab


class ControllerTests(unittest.TestCase):
    def test_crisp_thresholds_include_boundary(self):
        self.assertEqual(lab.crisp_speed(23.9, 69), 20)
        self.assertEqual(lab.crisp_speed(24, 69), 50)
        self.assertEqual(lab.crisp_speed(29.9, 60), 50)
        self.assertEqual(lab.crisp_speed(30, 60), 75)
        self.assertEqual(lab.crisp_speed(30, 70), 90)

    def test_controllers_reject_invalid_conditions(self):
        for controller in (
            lab.crisp_speed, lab.fuzzy_inference, lab.reference_speed, lab.NeuralController().speed,
        ):
            for inputs in ((17.9, 50), (42.1, 50), (30, -1), (30, 101), (float("nan"), 50)):
                with self.subTest(controller=controller, inputs=inputs):
                    with self.assertRaises(ValueError):
                        controller(*inputs)

    def test_fuzzy_rule_aggregation_and_centroid(self):
        corner = lab.fuzzy_inference(42, 100)
        self.assertEqual(corner.strongest, (1, 2, 2, 4))
        self.assertAlmostEqual(corner.speed, 92)

        overlap = lab.fuzzy_inference(36, 75)
        self.assertEqual(overlap.firing, ((0, 0, 0), (0, 0.5, 0.5), (0, 0.5, 0.5)))
        self.assertEqual(overlap.aggregate[:26], (0,) * 26)
        self.assertEqual(overlap.aggregate[38:], (0.5,) * 63)
        self.assertAlmostEqual(overlap.speed, 65.7856730214)
        self.assertEqual(lab.fuzzy_speed(36, 75), (overlap.speed, overlap.strongest))

    def test_hidden_weight_update_matches_numerical_gradient(self):
        model = lab.NeuralController()
        sample = (0.4, 0.7, 0.6)
        weights = model.hidden[0]
        initial = weights[0]
        epsilon = 1e-6
        weights[0] = initial + epsilon
        upper = model.mean_squared_error([sample]) / 2
        weights[0] = initial - epsilon
        lower = model.mean_squared_error([sample]) / 2
        weights[0] = initial
        gradient = (upper - lower) / (2 * epsilon)
        with patch.object(lab, "EPOCHS", 1):
            model.train(iter([sample]), 0.08)
        self.assertAlmostEqual(weights[0], initial - 0.08 * gradient, places=8)

    def test_experiment_starts_equal_and_uses_disjoint_data(self):
        self.assertEqual(len(lab.TRAINING_CASES), 143)
        self.assertEqual(len(lab.HELD_OUT_CASES), 120)
        self.assertTrue(set(lab.TRAINING_CASES).isdisjoint(lab.HELD_OUT_CASES))
        _, losses, errors = lab.run_experiment()
        self.assertEqual(losses[0][0], losses[1][0])
        for history in losses:
            self.assertEqual(len(history), lab.EPOCHS + 1)
            self.assertLess(history[-1], history[0])
        for error in errors[2:]:
            self.assertLess(error, 1.1)
