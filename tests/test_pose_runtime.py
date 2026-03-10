import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TrainRuntimeTests(unittest.TestCase):
    def test_build_dataloader_kwargs_respects_worker_settings(self):
        from utils.train_runtime import build_dataloader_kwargs

        no_worker_kwargs = build_dataloader_kwargs(
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        self.assertEqual(no_worker_kwargs['num_workers'], 0)
        self.assertNotIn('persistent_workers', no_worker_kwargs)
        self.assertNotIn('prefetch_factor', no_worker_kwargs)

        worker_kwargs = build_dataloader_kwargs(
            num_workers=2,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=3,
        )
        self.assertEqual(worker_kwargs['persistent_workers'], True)
        self.assertEqual(worker_kwargs['prefetch_factor'], 3)
        self.assertEqual(worker_kwargs['drop_last'], False)

    def test_epoch_loss_tracker_uses_weighted_average(self):
        from utils.train_runtime import EpochLossTracker

        tracker = EpochLossTracker()
        tracker.update(torch.tensor(2.0), {'loss_kpt': torch.tensor(4.0)}, weight=2)
        tracker.update(torch.tensor(6.0), {'loss_kpt': torch.tensor(10.0)}, weight=1)
        summary = tracker.summary()

        self.assertAlmostEqual(summary['loss'], (2.0 * 2 + 6.0) / 3)
        self.assertAlmostEqual(summary['loss_dict']['loss_kpt'], (4.0 * 2 + 10.0) / 3)

    def test_maybe_compile_model_falls_back_to_eager(self):
        from utils.train_runtime import maybe_compile_model

        model = nn.Linear(4, 4)
        with mock.patch.object(torch, 'compile', side_effect=RuntimeError('boom')):
            compiled = maybe_compile_model(model, compile_enabled=True)
        self.assertIs(compiled, model)

    def test_setcriterion_epoch_metrics_are_aggregated(self):
        from utils.loss import SetCriterion

        criterion = SetCriterion(
            num_logitsclass=1,
            matcher=None,
            weight_dict={},
            logits_eos_coef=0.1,
            losses=[],
            num_classes=3,
        )

        criterion._update_metric('MPJPE', torch.tensor([1.0, 3.0]))
        criterion.correct_num = 3
        criterion.class_count = 4
        criterion._update_conf_matrix(torch.tensor([0, 2]), torch.tensor([1, 2]))

        summary = criterion.get_epoch_metrics()
        self.assertEqual(summary['MPJPE'], 2.0)
        self.assertEqual(summary['accuracy'], 75.0)
        self.assertEqual(criterion.conf_matrix[1][0], 1)
        self.assertEqual(criterion.conf_matrix[2][2], 1)


class ConfigFlagTests(unittest.TestCase):
    def _load_config(self, module_name, extra_argv):
        sys.modules.pop(module_name, None)
        argv = [f'{module_name}.py'] + extra_argv
        with mock.patch.object(sys, 'argv', argv):
            return importlib.import_module(module_name)

    def test_backbone_pretrain_flags_parse_consistently(self):
        for module_name in ('config', 'config_mamba', 'config_attn_bilstm'):
            default_module = self._load_config(module_name, [])
            self.assertTrue(default_module.args.backbone_pretrain)

            disabled_module = self._load_config(module_name, ['--no_backbone_pretrain'])
            self.assertFalse(disabled_module.args.backbone_pretrain)


if __name__ == '__main__':
    unittest.main()
