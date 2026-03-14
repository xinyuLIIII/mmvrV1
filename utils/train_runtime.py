from contextlib import nullcontext
import os
import time

import torch


def resolve_device(device_name):
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device requested but not available: {device_name}')
    return device


def configure_runtime(device, matmul_precision='high'):
    torch.backends.cudnn.benchmark = device.type == 'cuda'
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(matmul_precision)


def get_amp_settings(device, amp_requested):
    amp_enabled = bool(amp_requested and device.type == 'cuda')
    amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    return amp_enabled, amp_device_type


def create_grad_scaler(device_type, enabled):
    return torch.amp.GradScaler(device_type, enabled=enabled)


def autocast_context(device_type, enabled):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def maybe_compile_model(model, compile_enabled, mode='reduce-overhead'):
    if not compile_enabled:
        return model
    if not hasattr(torch, 'compile'):
        raise RuntimeError('torch.compile is unavailable in the current PyTorch build.')
    try:
        compiled_model = torch.compile(model, mode=mode)
    except Exception as exc:
        print(f'torch.compile setup failed ({exc}). Falling back to eager mode.')
        return model
    print(f'torch.compile enabled with mode={mode}.')
    return compiled_model


def unwrap_model(model):
    return getattr(model, '_orig_mod', model)


def ensure_experiment_dirs(model_name, base_dir='./experiments'):
    for dirname in ('conf_matrix', 'weights', 'savept', 'param'):
        os.makedirs(os.path.join(base_dir, dirname, model_name), exist_ok=True)


def build_dataloader_kwargs(num_workers, pin_memory, shuffle, drop_last, persistent_workers, prefetch_factor):
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'shuffle': shuffle,
        'drop_last': drop_last,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return loader_kwargs


def create_pose_lr_scheduler(optimizer, args):
    scheduler_name = getattr(args, 'lr_scheduler', 'step')
    if scheduler_name == 'none':
        return None
    if scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )
    if scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.5)
    raise ValueError(f'Unsupported lr scheduler: {scheduler_name}')


def step_pose_lr_scheduler(scheduler, scheduler_name, monitor_value=None):
    if scheduler is None:
        return
    if scheduler_name == 'plateau':
        if monitor_value is None:
            raise ValueError('monitor_value is required for ReduceLROnPlateau.')
        scheduler.step(monitor_value)
        return
    scheduler.step()


def get_pose_monitor_value(loss_summary, metric_summary, monitor_name):
    if monitor_name == 'loss':
        return float(loss_summary.get('loss', 0.0))
    return float(metric_summary.get(monitor_name, 0.0))


def get_optimizer_lr(optimizer):
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]['lr'])


def update_early_stopping_state(best_value, current_value, bad_epochs, patience):
    improved = best_value is None or current_value < best_value
    if improved:
        return current_value, 0, False, True

    bad_epochs += 1
    should_stop = patience > 0 and bad_epochs >= patience
    return best_value, bad_epochs, should_stop, False


def move_targets_to_device(targets, device):
    return [{key: value.to(device, non_blocking=True) for key, value in item.items()} for item in targets]


class EpochLossTracker:
    def __init__(self):
        self.total_weight = 0
        self.total_loss_sum = None
        self.loss_sums = {}

    def update(self, total_loss, loss_dict, weight=1):
        if weight <= 0:
            return
        self.total_weight += weight
        total_loss_sum = total_loss.detach().to(dtype=torch.float64) * weight
        if self.total_loss_sum is None:
            self.total_loss_sum = total_loss_sum
        else:
            self.total_loss_sum = self.total_loss_sum + total_loss_sum
        for key, value in loss_dict.items():
            loss_value_sum = value.detach().to(dtype=torch.float64) * weight
            if key not in self.loss_sums:
                self.loss_sums[key] = loss_value_sum
            else:
                self.loss_sums[key] = self.loss_sums[key] + loss_value_sum

    def summary(self):
        if self.total_weight == 0:
            return {'loss': 0.0, 'loss_dict': {}, 'runtime': {}}
        weight = float(self.total_weight)
        return {
            'loss': float((self.total_loss_sum / weight).cpu().item()),
            'loss_dict': {
                key: float((value / weight).cpu().item()) for key, value in sorted(self.loss_sums.items())
            },
            'runtime': {},
        }


def run_pose_epoch(
    model,
    criterion,
    data_loader,
    device,
    amp_device_type,
    amp_enabled,
    optimizer=None,
    scaler=None,
    grad_accum_steps=1,
    clip_max_norm=0.0,
    mem_stats_active=False,
):
    is_train = optimizer is not None
    if is_train and scaler is None:
        raise ValueError('GradScaler is required when optimizer is provided.')

    loss_tracker = EpochLossTracker()
    mem_stats = None
    if mem_stats_active:
        from utils import misc

        mem_stats = misc.MemStatsAccumulator()

    if is_train:
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
        criterion.eval()

    train_num_steps = len(data_loader)
    grad_accum_steps = max(1, grad_accum_steps)
    total_data_time = 0.0
    total_step_time = 0.0
    total_samples = 0

    from tqdm import tqdm

    next_data_start = time.perf_counter()
    for step, (samples, imu, target) in enumerate(tqdm(data_loader)):
        data_ready = time.perf_counter()
        total_data_time += data_ready - next_data_start
        step_start = data_ready
        samples = samples.to(device, non_blocking=True)
        imu = imu.to(device, non_blocking=True)
        target = move_targets_to_device(target, device)
        batch_size = samples.shape[0]
        total_samples += int(batch_size)
        grad_context = nullcontext() if is_train else torch.no_grad()
        with grad_context:
            with autocast_context(amp_device_type, enabled=amp_enabled):
                out = model(samples, imu)
                if mem_stats_active and 'pose_memory' in out:
                    mem_stats.update(out['pose_memory'])
                loss_dict = criterion(out, target)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[key] * weight_dict[key] for key in loss_dict.keys() if key in weight_dict)
        loss_tracker.update(losses, loss_dict, weight=batch_size)
        if is_train:
            losses_for_backward = losses / grad_accum_steps
            scaler.scale(losses_for_backward).backward()
            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == train_num_steps)
            if should_step:
                if clip_max_norm and clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_step_time += time.perf_counter() - step_start
        next_data_start = time.perf_counter()

    runtime_summary = {
        'data_time': total_data_time / train_num_steps if train_num_steps else 0.0,
        'step_time': total_step_time / train_num_steps if train_num_steps else 0.0,
        'samples_per_sec': total_samples / total_step_time if total_step_time > 0 else 0.0,
    }

    mem_summary = mem_stats.summary() if mem_stats is not None else None
    loss_summary = loss_tracker.summary()
    loss_summary['runtime'] = runtime_summary
    return loss_summary, criterion.get_epoch_metrics(), mem_summary
