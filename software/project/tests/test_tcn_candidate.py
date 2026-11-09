"""Synthetic structure checks only; no dataset, training run or model-score claim."""

import copy
import io
import sys
from pathlib import Path
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tcn_candidate import (CausalConv1d, build_model, load_config,
                           model_summary, validate_config)


class TcnCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.config = load_config()
        self.net = build_model(self.config).eval()

    def test_review_draft_cannot_inherit_training_approval(self):
        self.assertIs(self.config['approved_for_formal_training'], False)
        for key in ('approved_for_formal_training', 'inherit_training_approval', 'resume_old_cnn_checkpoint'):
            changed = copy.deepcopy(self.config)
            changed[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed['formal_training_budget']['maximum_runs'] = 1
        with self.assertRaises(ValueError):
            validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed['experiment'] = 'E01'
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_candidate_rejects_unreviewed_contract_drift(self):
        for key, value in (('train_batch_size', 128), ('learning_rate', .003),
                           ('seed', 7), ('early_stop_patience', 20),
                           ('validation_split', 'final_test')):
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(changed)
        for key, value in (('hidden_channels', 64), ('padding', 'symmetric'),
                           ('readout', 'temporal_mean'), ('weight_norm', True)):
            changed = copy.deepcopy(self.config)
            changed['model'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(changed)
        for key, value in (('horizon', 100), ('threshold', .0001),
                           ('additional_factors', True), ('adaptive_input_normalization', True)):
            changed = copy.deepcopy(self.config)
            changed['data_contract'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(changed)

    def test_model_geometry_parameter_count_and_summary(self):
        with torch.inference_mode():
            self.assertEqual(tuple(self.net(torch.zeros(3, 40, 100)).shape), (3, 3))
            self.assertEqual(tuple(self.net.forward_features(torch.zeros(1, 40, 100)).shape), (1, 32, 100))
        summary = model_summary(self.config)
        self.assertEqual(sum(p.numel() for p in self.net.parameters()), 33219)
        self.assertEqual(summary['parameter_count'], 33219)
        self.assertEqual(summary['theoretical_receptive_field_records'], 125)
        self.assertEqual(summary['dense_macs_per_window'], 3276896)
        self.assertIs(summary['predictive_quality_verified'], False)
        self.assertIs(summary['hardware_verified'], False)

    def test_exact_window_contract_and_no_look_outside_100_records(self):
        for shape in ((40, 100), (1, 100, 40), (1, 40, 99), (1, 40, 101), (0, 40, 100)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.net(torch.zeros(shape))
        conv = CausalConv1d(1, 1, 3, 16).eval()
        with torch.no_grad():
            conv.conv.weight.fill_(1)
            conv.conv.bias.zero_()
        x = torch.arange(1, 101, dtype=torch.float32).reshape(1, 1, 100)
        expected = torch.tensor([sum(float(x[0, 0, t - lag]) for lag in (0, 16, 32) if t >= lag)
                                 for t in range(100)]).reshape(1, 1, 100)
        with torch.inference_mode():
            torch.testing.assert_close(conv(x), expected, rtol=0, atol=0)

    def test_future_within_window_cannot_change_prefix_features(self):
        x = torch.randn(2, 40, 100)
        with torch.inference_mode():
            reference = self.net.forward_features(x)
            for last_past_index in (0, 1, 17, 50, 98):
                changed = x.clone()
                changed[:, :, last_past_index + 1:] += 50 * torch.randn_like(changed[:, :, last_past_index + 1:])
                actual = self.net.forward_features(changed)
                with self.subTest(last_past_index=last_past_index):
                    torch.testing.assert_close(actual[:, :, :last_past_index + 1],
                                               reference[:, :, :last_past_index + 1], rtol=0, atol=0)

    def test_lagwise_operation_matches_original_dilated_conv_and_gradients(self):
        for dilation in (1, 2, 4, 8, 16):
            actual = CausalConv1d(7, 5, 3, dilation)
            reference = nn.Conv1d(7, 5, 3, dilation=dilation, bias=True)
            reference.load_state_dict(actual.conv.state_dict())
            x = torch.randn(2, 7, 100, requires_grad=True)
            rx = x.detach().clone().requires_grad_(True)
            out = actual(x)
            expected = reference(nn.functional.pad(rx, (2 * dilation, 0)))
            upstream = torch.randn_like(out)
            out.backward(upstream)
            expected.backward(upstream)
            with self.subTest(dilation=dilation):
                torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(x.grad, rx.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual.conv.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(actual.conv.bias.grad, reference.bias.grad, rtol=1e-5, atol=1e-5)
                self.assertEqual(list(actual.state_dict()), ['conv.weight', 'conv.bias'])
                self.assertEqual(sum(p.numel() for p in actual.parameters()), 110)

    def test_last_time_readout_has_path_from_first_record(self):
        for layer in self.net.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                with torch.no_grad():
                    layer.weight.fill_(.05)
                    layer.bias.zero_()
        x = torch.ones(1, 40, 100, requires_grad=True)
        features = self.net.forward_features(x)
        logits = self.net(x)
        torch.testing.assert_close(logits, self.net.classifier(features[:, :, -1]), rtol=0, atol=0)
        logits.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[0, :, 0].abs().sum()), 0)
        self.assertGreater(float(x.grad[0, :, -1].abs().sum()), 0)

    def test_synthetic_cross_entropy_backward_is_finite_without_updates(self):
        x = torch.randn(3, 40, 100)
        before = {name: p.detach().clone() for name, p in self.net.named_parameters()}
        loss = nn.functional.cross_entropy(self.net(x), torch.tensor([0, 1, 2]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in self.net.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_state_and_scripted_export_round_trip_preserve_scores_and_shape_gate(self):
        x = torch.randn(3, 40, 100)
        state_buffer = io.BytesIO()
        torch.save(self.net.state_dict(), state_buffer)
        state_buffer.seek(0)
        reloaded = build_model(self.config).eval()
        reloaded.load_state_dict(torch.load(state_buffer, weights_only=True))
        script_buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(self.net), script_buffer)
        script_buffer.seek(0)
        exported = torch.jit.load(script_buffer).eval()
        with torch.inference_mode():
            expected = self.net(x)
            torch.testing.assert_close(reloaded(x), expected, rtol=0, atol=0)
            torch.testing.assert_close(exported(x), expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(exported(x).argmax(1), expected.argmax(1)))
        with self.assertRaises((ValueError, torch.jit.Error)):
            exported(torch.zeros(1, 40, 101))


if __name__ == '__main__':
    unittest.main()
