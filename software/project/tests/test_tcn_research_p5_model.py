"""Generated-input P5 model checks; no records, checkpoint or optimizer use.

Executable directly to retain exact construction/forward/backward counters.
The optional GPU test is skipped only if CUDA is unavailable and is explicit.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

COUNTS = {'model_constructions': 0, 'model_forward_calls': 0,
          'additional_forward_features_calls': 0, 'backward_calls': 0,
          'optimizer_constructions': 0, 'optimizer_steps': 0,
          'formal_training_runs': 0, 'market_records_or_weights_read': False,
          'checkpoint_serializations': 0}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, factory, candidate
        import torch
        import tcn_research_p5_model as factory
        import tcn_candidate as candidate
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    def model(self, wn=False, seed=741):
        torch.manual_seed(seed)
        model = factory.build_model(wn)
        COUNTS['model_constructions'] += 1
        return model

    def forward(self, model, x):
        COUNTS['model_forward_calls'] += 1
        return model(x)

    def inputs(self, count=2):
        generator = torch.Generator().manual_seed(2041)
        return torch.randn(count, 40, 100, generator=generator)

    def temporal(self, model):
        return [layer for layer in model.modules() if isinstance(layer, candidate.CausalConv1d)]

    def test_01_import_does_not_import_ML(self):
        code = ("import sys;sys.path.insert(0," + repr(str(ROOT / 'scripts')) + ");"
                "import tcn_research_p5_model;"
                "assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules;"
                "assert 'tcn_candidate' not in sys.modules")
        completed = subprocess.run([sys.executable, '-I', '-c', code],
                                   capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_02_reject_non_boolean_before_ML(self):
        for value in (None, 0, 1, 'false', [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                factory.build_model(value)

    def test_03_baseline_exact_frozen_class_and_parameters(self):
        model = self.model(False)
        self.assertIs(type(model), candidate.TemporalConvCandidate)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 33219)
        self.assertEqual(len(list(model.parameters())), 24)
        self.assertEqual(len(self.temporal(model)), 10)
        self.assertFalse(any('parametrizations' in name for name in model.state_dict()))

    def test_04_only_ten_temporal_weights_are_normalized(self):
        model = self.model(True)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 33539)
        self.assertEqual(len(list(model.parameters())), 34)
        originals = [name for name in model.state_dict() if 'parametrizations.weight.original0' in name]
        self.assertEqual(len(originals), 10)
        self.assertTrue(all('.temporal.' in name for name in originals))
        self.assertFalse(any('parametrizations' in name for name in model.features[0].shortcut.state_dict()))
        self.assertEqual(set(model.classifier.state_dict()), {'weight', 'bias'})
        for layer in self.temporal(model):
            self.assertEqual(tuple(layer.conv.parametrizations.weight.original0.shape), (32, 1, 1))

    def test_05_effective_initial_weights_preserve_same_seed(self):
        baseline, normalized = self.model(False), self.model(True)
        for left, right in zip(self.temporal(baseline), self.temporal(normalized)):
            torch.testing.assert_close(left.conv.weight, right.conv.weight, rtol=2e-6, atol=2e-8)
            torch.testing.assert_close(left.conv.bias, right.conv.bias, rtol=0, atol=0)
        torch.testing.assert_close(baseline.features[0].shortcut.weight,
                                   normalized.features[0].shortcut.weight, rtol=0, atol=0)
        torch.testing.assert_close(baseline.classifier.weight, normalized.classifier.weight, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(self.forward(baseline, self.inputs()),
                                       self.forward(normalized, self.inputs()), rtol=2e-5, atol=2e-6)

    def test_06_lagwise_weight_reads_refresh_after_update(self):
        model = self.model(True)
        layer = self.temporal(model)[-1]
        self.assertGreater(layer.conv.dilation[0], 1)
        x = self.inputs()
        with torch.no_grad():
            weight_before = layer.conv.weight.clone()
            before = self.forward(model, x).clone()
            layer.conv.parametrizations.weight.original0.mul_(1.7)
            weight_after = layer.conv.weight.clone()
            after = self.forward(model, x).clone()
            repeat = self.forward(model, x).clone()
        torch.testing.assert_close(weight_after, weight_before * 1.7, rtol=2e-6, atol=2e-8)
        self.assertGreater(float((after - before).abs().max()), 1e-8)
        torch.testing.assert_close(after, repeat, rtol=0, atol=0)

    def test_07_gradients_reach_every_weight_normalized_parameter(self):
        model = self.model(True)
        output = self.forward(model, self.inputs(4))
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 2, 0]))
        loss.backward(); COUNTS['backward_calls'] += 1
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.norm()), 0.0, name)
        model.zero_grad(set_to_none=True)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_08_forward_features_causality(self):
        model = self.model(True).eval()
        x = self.inputs()
        altered = x.clone(); altered[:, :, 61:] += 31
        with torch.inference_mode():
            COUNTS['additional_forward_features_calls'] += 2
            first, second = model.forward_features(x), model.forward_features(altered)
        torch.testing.assert_close(first[:, :, :61], second[:, :, :61], rtol=0, atol=0)
        self.assertGreater(float((first[:, :, 61:] - second[:, :, 61:]).abs().max()), 1e-5)

    def test_09_strict_state_dict_roundtrip(self):
        original, restored = self.model(True), self.model(True, seed=159)
        frozen = {name: value.detach().clone() for name, value in original.state_dict().items()}
        restored.load_state_dict(frozen, strict=True)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(self.forward(original, self.inputs()),
                                       self.forward(restored, self.inputs()), rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in original.parameters()))

    def test_10_reject_cross_architecture_state(self):
        baseline, normalized = self.model(False), self.model(True)
        with self.assertRaises(RuntimeError): normalized.load_state_dict(baseline.state_dict(), strict=True)
        with self.assertRaises(RuntimeError): baseline.load_state_dict(normalized.state_dict(), strict=True)

    def test_11_CPU_GPU_weight_norm_consistency(self):
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable; no GPU result claimed')
        cpu, gpu = self.model(True).eval(), self.model(True, seed=71).eval().cuda()
        gpu.load_state_dict(cpu.state_dict(), strict=True)
        torch.cuda.synchronize()
        with torch.inference_mode():
            expected = self.forward(cpu, self.inputs())
            actual = self.forward(gpu, self.inputs().cuda()).cpu()
        torch.cuda.synchronize()
        maximum = float((actual - expected).abs().max())
        print(json.dumps({'CPU_GPU_max_abs_error': maximum}, ensure_ascii=True), flush=True)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertTrue(bool(torch.equal(actual.argmax(1), expected.argmax(1))))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests))
    print(json.dumps({'operations': COUNTS, 'tests_run': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'skipped': len(result.skipped), 'synthetic_only': True,
                      'formal_accuracy_measured': False}, ensure_ascii=True), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
