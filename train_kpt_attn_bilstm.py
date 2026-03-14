from dataset.datasets import build_dataset
from torch.utils.data import DataLoader
import torch
import config_attn_bilstm as config
from tensorboardX import SummaryWriter
from models.mmVR_AttnBiLSTM import build_model
from utils import misc
from utils.train_runtime import (
    build_dataloader_kwargs,
    configure_runtime,
    create_pose_lr_scheduler,
    create_grad_scaler,
    ensure_experiment_dirs,
    get_optimizer_lr,
    get_amp_settings,
    get_pose_monitor_value,
    maybe_compile_model,
    run_pose_epoch,
    resolve_device,
    step_pose_lr_scheduler,
    update_early_stopping_state,
    unwrap_model,
)
import os
import json


def _write_mem_stats(writer, summary, split, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'pose_memory_{split}_epoch_{epoch + 1}.json')
    with open(path, 'w') as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    for key, value in summary.items():
        writer.add_scalar(f'mem/pose/{key}', value, epoch + 1)


if __name__ == '__main__':
    args = config.args
    save_path = './experiments/'
    model_name = 'train_attn_bilstm'
    device = resolve_device(args.device)
    configure_runtime(device, args.matmul_precision)
    if model_name != 'test':
        ensure_experiment_dirs(model_name)
    amp_enabled, amp_device_type = get_amp_settings(device, args.amp)
    if args.amp and not amp_enabled:
        print(f'AMP requested, but device {device} does not support CUDA AMP. Running without AMP.')
    else:
        print(f'AMP enabled: {amp_enabled}')

    train_dataset = build_dataset(args.dataset_root, args.mode, args=args)
    train_loader_kwargs = build_dataloader_kwargs(
        args.num_workers,
        args.pin_memory,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader_kwargs = build_dataloader_kwargs(
        args.num_workers,
        args.pin_memory,
        shuffle=False,
        drop_last=False,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=misc.collate_fn,
        **train_loader_kwargs,
    )
    train_size = int(train_dataset.__len__())

    test_dataset = build_dataset(args.dataset_root, False, args=args)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.val_batch_size,
        collate_fn=misc.collate_fn,
        **test_loader_kwargs,
    )
    test_size = int(test_dataset.__len__())
    print('model_name: ', model_name)
    print('train_size: ', train_size)
    print('test_size: ', test_size)
    model, criterion = build_model(args)
    model = model.to(device)
    grad_accum_steps = max(1, args.grad_accum_steps)
    if grad_accum_steps != args.grad_accum_steps:
        print(f'Adjusted grad_accum_steps from {args.grad_accum_steps} to {grad_accum_steps}.')
    if grad_accum_steps > 1:
        print(f'Using gradient accumulation: {grad_accum_steps} steps.')

    base_batch_size = 32
    effective_batch_size = args.batch_size * grad_accum_steps
    effective_lr = args.lr
    if effective_batch_size != base_batch_size and abs(args.lr - 1e-4) < 1e-12:
        effective_lr = args.lr * effective_batch_size / base_batch_size
        print(
            f'Auto-scaled lr from {args.lr:.8f} to {effective_lr:.8f} '
            f'for effective_batch_size={effective_batch_size} '
            f'(batch_size={args.batch_size}, grad_accum_steps={grad_accum_steps}, '
            f'base_batch_size={base_batch_size}).'
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    model = maybe_compile_model(model, args.compile)
    lr_scheduler = create_pose_lr_scheduler(optimizer, args)
    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    mem_stats_enabled = args.mem_stats
    mem_stats_every = max(1, args.mem_stats_every)
    mem_stats_dir = args.mem_stats_save
    best_mpjpe_train = 1000.0
    best_acc_train = 0.0
    best_mpjpe_test = 1000.0
    best_acc_test = 0.0
    best_early_stop_mpjpe = None
    early_stop_bad_epochs = 0
    temp_test = 0
    scaler = create_grad_scaler(amp_device_type, enabled=amp_enabled)

    for epoch in range(args.epoch):
        current_lr = get_optimizer_lr(optimizer)
        print(f'epoch {epoch + 1} lr: {current_lr:.8f}')
        train_logger.info({'epoch': epoch + 1, 'lr': round(current_lr, 10)})
        train_writer.add_scalar('LR/train', current_lr, epoch + 1)
        test_writer.add_scalar('LR/test', current_lr, epoch + 1)
        misc.criterion_init(criterion)
        mem_stats_active = mem_stats_enabled and (epoch % mem_stats_every == 0)
        train_loss_summary, train_metric_summary, train_mem_summary = run_pose_epoch(
            model,
            criterion,
            train_loader,
            device,
            amp_device_type,
            amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
            clip_max_norm=args.clip_max_norm,
            mem_stats_active=mem_stats_active,
        )
        best_mpjpe_train, best_acc_train, _, is_save = misc.log_save(
            model_name,
            train_writer,
            train_logger,
            epoch,
            train_loss_summary,
            train_metric_summary,
            criterion,
            best_mpjpe_train,
            best_acc_train,
            mode=True,
        )

        misc.criterion_init(criterion)
        test_loss_summary, test_metric_summary, test_mem_summary = run_pose_epoch(
            model,
            criterion,
            test_loader,
            device,
            amp_device_type,
            amp_enabled,
            mem_stats_active=mem_stats_active,
        )
        best_mpjpe_test, best_acc_test, temp_test, is_save = misc.log_save(
            model_name,
            test_writer,
            test_logger,
            epoch,
            test_loss_summary,
            test_metric_summary,
            criterion,
            best_mpjpe_test,
            best_acc_test,
            mode=False,
            temp_test=temp_test,
        )
        scheduler_monitor = get_pose_monitor_value(
            test_loss_summary,
            test_metric_summary,
            args.plateau_metric,
        )
        step_pose_lr_scheduler(lr_scheduler, args.lr_scheduler, scheduler_monitor)
        current_test_mpjpe = get_pose_monitor_value(
            test_loss_summary,
            test_metric_summary,
            'MPJPE',
        )
        best_early_stop_mpjpe, early_stop_bad_epochs, should_stop, improved = update_early_stopping_state(
            best_early_stop_mpjpe,
            current_test_mpjpe,
            early_stop_bad_epochs,
            args.early_stop_patience,
        )
        print(
            f'validation monitor ({args.plateau_metric}): {scheduler_monitor:.4f}; '
            f'early_stop_mpjpe: {current_test_mpjpe:.4f}; '
            f'bad_epochs: {early_stop_bad_epochs}'
        )
        test_logger.info({
            'epoch': epoch + 1,
            'lr': round(get_optimizer_lr(optimizer), 10),
            'scheduler_monitor': round(scheduler_monitor, 4),
            'scheduler_monitor_name': args.plateau_metric,
            'early_stop_mpjpe': round(current_test_mpjpe, 4),
            'early_stop_bad_epochs': early_stop_bad_epochs,
            'early_stop_improved': improved,
        })
        if mem_stats_active:
            _write_mem_stats(train_writer, train_mem_summary, 'train', epoch, mem_stats_dir)
            _write_mem_stats(test_writer, test_mem_summary, 'test', epoch, mem_stats_dir)
        if is_save:
            torch.save(unwrap_model(model).state_dict(), f'./experiments/param/{model_name}/best_test_mpjpe.pth')
        if should_stop:
            print(
                f'Early stopping triggered at epoch {epoch + 1}: '
                f'validation MPJPE did not improve for {early_stop_bad_epochs} epochs.'
            )
            break

    train_writer.close()
    test_writer.close()
    print('model_name: ', model_name)
