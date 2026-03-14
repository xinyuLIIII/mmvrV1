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
        self.assertEqual(summary['runtime'], {})

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
        self.assertEqual(summary['correct_num'], 3)
        self.assertEqual(criterion.conf_matrix[1][0], 1)
        self.assertEqual(criterion.conf_matrix[2][2], 1)

    def test_run_pose_epoch_records_runtime_summary(self):
        from utils.train_runtime import create_grad_scaler, run_pose_epoch

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, samples, imu):
                return {'prediction': (samples + imu) * self.scale}

        class DummyCriterion(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_dict = {'loss_main': 1.0}

            def forward(self, outputs, targets):
                loss = outputs['prediction'].mean()
                return {'loss_main': loss}

            def get_epoch_metrics(self):
                return {'MPJPE': 0.0, 'accuracy': 0.0}

        samples = torch.ones(4, 2, 3)
        imu = torch.ones(4, 2, 3)
        targets = [{'dummy': torch.tensor(1)} for _ in range(4)]
        data_loader = [
            (samples[:2], imu[:2], targets[:2]),
            (samples[2:], imu[2:], targets[2:]),
        ]

        model = DummyModel()
        loss_summary, metric_summary, mem_summary = run_pose_epoch(
            model,
            DummyCriterion(),
            data_loader,
            torch.device('cpu'),
            'cpu',
            False,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            scaler=create_grad_scaler('cpu', enabled=False),
        )

        self.assertIn('runtime', loss_summary)
        self.assertGreaterEqual(loss_summary['runtime']['data_time'], 0.0)
        self.assertGreater(loss_summary['runtime']['samples_per_sec'], 0.0)
        self.assertEqual(metric_summary['MPJPE'], 0.0)
        self.assertIsNone(mem_summary)

    def test_create_pose_lr_scheduler_plateau_and_step(self):
        from utils.train_runtime import create_pose_lr_scheduler

        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        plateau_args = mock.Mock(
            lr_scheduler='plateau',
            lr_drop=200,
            plateau_factor=0.5,
            plateau_patience=2,
            plateau_min_lr=1e-6,
        )
        scheduler = create_pose_lr_scheduler(optimizer, plateau_args)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        step_args = mock.Mock(
            lr_scheduler='step',
            lr_drop=7,
            plateau_factor=0.5,
            plateau_patience=2,
            plateau_min_lr=1e-6,
        )
        scheduler = create_pose_lr_scheduler(optimizer, step_args)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 7)

        none_args = mock.Mock(
            lr_scheduler='none',
            lr_drop=7,
            plateau_factor=0.5,
            plateau_patience=2,
            plateau_min_lr=1e-6,
        )
        self.assertIsNone(create_pose_lr_scheduler(optimizer, none_args))

    def test_step_pose_lr_scheduler_uses_validation_metric_for_plateau(self):
        from utils.train_runtime import create_pose_lr_scheduler, step_pose_lr_scheduler

        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        args = mock.Mock(
            lr_scheduler='plateau',
            lr_drop=200,
            plateau_factor=0.5,
            plateau_patience=1,
            plateau_min_lr=1e-6,
        )
        scheduler = create_pose_lr_scheduler(optimizer, args)

        step_pose_lr_scheduler(scheduler, args.lr_scheduler, 1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1)
        step_pose_lr_scheduler(scheduler, args.lr_scheduler, 1.1)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1)
        step_pose_lr_scheduler(scheduler, args.lr_scheduler, 1.2)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.05)

    def test_get_pose_monitor_value_supports_metrics_and_loss(self):
        from utils.train_runtime import get_pose_monitor_value

        loss_summary = {'loss': 0.75}
        metric_summary = {'MPJPE': 12.5, 'MPJDLE': 6.25}

        self.assertEqual(get_pose_monitor_value(loss_summary, metric_summary, 'MPJPE'), 12.5)
        self.assertEqual(get_pose_monitor_value(loss_summary, metric_summary, 'MPJDLE'), 6.25)
        self.assertEqual(get_pose_monitor_value(loss_summary, metric_summary, 'loss'), 0.75)

    def test_update_early_stopping_state_tracks_best_and_bad_epochs(self):
        from utils.train_runtime import update_early_stopping_state

        best_value, bad_epochs, should_stop, improved = update_early_stopping_state(
            best_value=None,
            current_value=10.0,
            bad_epochs=0,
            patience=3,
        )
        self.assertEqual(best_value, 10.0)
        self.assertEqual(bad_epochs, 0)
        self.assertFalse(should_stop)
        self.assertTrue(improved)

        best_value, bad_epochs, should_stop, improved = update_early_stopping_state(
            best_value=best_value,
            current_value=11.0,
            bad_epochs=bad_epochs,
            patience=3,
        )
        self.assertEqual(best_value, 10.0)
        self.assertEqual(bad_epochs, 1)
        self.assertFalse(should_stop)
        self.assertFalse(improved)

        best_value, bad_epochs, should_stop, improved = update_early_stopping_state(
            best_value=best_value,
            current_value=9.5,
            bad_epochs=bad_epochs,
            patience=3,
        )
        self.assertEqual(best_value, 9.5)
        self.assertEqual(bad_epochs, 0)
        self.assertFalse(should_stop)
        self.assertTrue(improved)

        _, bad_epochs, should_stop, improved = update_early_stopping_state(
            best_value=best_value,
            current_value=9.7,
            bad_epochs=2,
            patience=3,
        )
        self.assertEqual(bad_epochs, 3)
        self.assertTrue(should_stop)
        self.assertFalse(improved)


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

    def test_attn_bilstm_defaults_enable_throughput_friendly_settings(self):
        module = self._load_config('config_attn_bilstm', [])

        self.assertEqual(module.args.val_batch_size, 8)
        self.assertEqual(module.args.num_workers, 8)
        self.assertEqual(module.args.prefetch_factor, 4)
        self.assertFalse(module.args.amp)
        self.assertTrue(module.args.persistent_workers)
        self.assertEqual(module.args.lr_scheduler, 'plateau')
        self.assertEqual(module.args.plateau_metric, 'MPJPE')
        self.assertEqual(module.args.plateau_factor, 0.5)
        self.assertEqual(module.args.plateau_patience, 5)
        self.assertEqual(module.args.plateau_min_lr, 1e-6)
        self.assertEqual(module.args.early_stop_patience, 10)

    def test_attn_bilstm_scheduler_flags_parse_consistently(self):
        module = self._load_config(
            'config_attn_bilstm',
            [
                '--lr_scheduler', 'step',
                '--plateau_metric', 'loss',
                '--plateau_factor', '0.2',
                '--plateau_patience', '3',
                '--plateau_min_lr', '1e-5',
                '--early_stop_patience', '0',
            ],
        )

        self.assertEqual(module.args.lr_scheduler, 'step')
        self.assertEqual(module.args.plateau_metric, 'loss')
        self.assertEqual(module.args.plateau_factor, 0.2)
        self.assertEqual(module.args.plateau_patience, 3)
        self.assertEqual(module.args.plateau_min_lr, 1e-5)
        self.assertEqual(module.args.early_stop_patience, 0)

    def test_stage1_configs_expose_cfar_flags(self):
        for module_name in ('config', 'config_mamba', 'config_attn_bilstm'):
            default_module = self._load_config(module_name, [])
            self.assertEqual(default_module.args.cfar_mode, 'none')
            self.assertEqual(default_module.args.cfar_soft_mode, 'subtract')
            self.assertTrue(default_module.args.cfar_split_halves)

            enabled_module = self._load_config(
                module_name,
                [
                    '--cfar_mode', 'os2d',
                    '--cfar_guard', '2',
                    '--cfar_train', '5',
                    '--cfar_rank_ratio', '0.6',
                    '--cfar_pfa', '0.1',
                    '--cfar_soft_mode', 'mask',
                    '--no_cfar_split_halves',
                ],
            )
            self.assertEqual(enabled_module.args.cfar_mode, 'os2d')
            self.assertEqual(enabled_module.args.cfar_guard, 2)
            self.assertEqual(enabled_module.args.cfar_train, 5)
            self.assertAlmostEqual(enabled_module.args.cfar_rank_ratio, 0.6)
            self.assertAlmostEqual(enabled_module.args.cfar_pfa, 0.1)
            self.assertEqual(enabled_module.args.cfar_soft_mode, 'mask')
            self.assertFalse(enabled_module.args.cfar_split_halves)


if __name__ == '__main__':
    unittest.main()
