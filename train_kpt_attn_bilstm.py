from dataset.datasets import build_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config_attn_bilstm as config
from tensorboardX import SummaryWriter
from models.mmVR_AttnBiLSTM import build_model
from utils import misc
from utils.train_runtime import (
    autocast_context,
    configure_runtime,
    create_grad_scaler,
    get_amp_settings,
    maybe_compile_model,
    resolve_device,
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
        if not os.path.exists('./experiments/param/' + model_name + '/'):
            os.makedirs('./experiments/conf_matrix/' + model_name)
            os.makedirs('./experiments/weights/' + model_name)
            os.makedirs('./experiments/savept/' + model_name)
            os.makedirs('./experiments/param/' + model_name)
    amp_enabled, amp_device_type = get_amp_settings(device, args.amp)
    if args.amp and not amp_enabled:
        print(f'AMP requested, but device {device} does not support CUDA AMP. Running without AMP.')
    else:
        print(f'AMP enabled: {amp_enabled}')

    train_dataset = build_dataset(args.dataset_root, args.mode)
    train_loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'shuffle': True,
        'drop_last': True,
    }
    test_loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'shuffle': False,
        'drop_last': True,
    }
    if args.num_workers > 0:
        train_loader_kwargs['persistent_workers'] = args.persistent_workers
        test_loader_kwargs['persistent_workers'] = args.persistent_workers
        train_loader_kwargs['prefetch_factor'] = args.prefetch_factor
        test_loader_kwargs['prefetch_factor'] = args.prefetch_factor

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=misc.collate_fn,
        **train_loader_kwargs,
    )
    train_size = int(train_dataset.__len__())

    test_dataset = build_dataset(args.dataset_root, False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.5)
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
    temp_test = 0
    scaler = create_grad_scaler(amp_device_type, enabled=amp_enabled)

    for epoch in range(args.epoch):
        misc.criterion_init(criterion)
        mem_stats_active = mem_stats_enabled and (epoch % mem_stats_every == 0)
        if mem_stats_active:
            train_mem_stats = misc.MemStatsAccumulator()
        loss_dict = {}
        losses = 0
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        train_num_steps = len(train_loader)
        for step, (samples, imu, target) in enumerate(tqdm(train_loader)):
            samples = samples.to(device, non_blocking=True)
            imu = imu.to(device, non_blocking=True)
            target = [{key: value.to(device, non_blocking=True) for key, value in item.items()} for item in target]
            with autocast_context(amp_device_type, enabled=amp_enabled):
                out = model(samples, imu)
                if mem_stats_active and 'pose_memory' in out:
                    train_mem_stats.update(out['pose_memory'])
                loss_dict = criterion(out, target)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[key] * weight_dict[key] for key in loss_dict.keys() if key in weight_dict)
            losses_for_backward = losses / grad_accum_steps
            scaler.scale(losses_for_backward).backward()
            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == train_num_steps)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        best_mpjpe_train, best_acc_train, _, is_save = misc.log_save(
            model_name,
            train_writer,
            train_logger,
            epoch,
            losses,
            loss_dict,
            criterion,
            train_size,
            best_mpjpe_train,
            best_acc_train,
            mode=True,
        )

        misc.criterion_init(criterion)
        if mem_stats_active:
            test_mem_stats = misc.MemStatsAccumulator()
        loss_dict = {}
        losses = 0
        model.eval()
        criterion.eval()
        for samples, imu, target in tqdm(test_loader):
            with torch.no_grad():
                target = [{key: value.to(device, non_blocking=True) for key, value in item.items()} for item in target]
                samples = samples.to(device, non_blocking=True)
                imu = imu.to(device, non_blocking=True)
                with autocast_context(amp_device_type, enabled=amp_enabled):
                    out = model(samples, imu)
                    if mem_stats_active and 'pose_memory' in out:
                        test_mem_stats.update(out['pose_memory'])
                    loss_dict = criterion(out, target)
                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[key] * weight_dict[key] for key in loss_dict.keys() if key in weight_dict)
        best_mpjpe_test, best_acc_test, temp_test, is_save = misc.log_save(
            model_name,
            test_writer,
            test_logger,
            epoch,
            losses,
            loss_dict,
            criterion,
            test_size,
            best_mpjpe_test,
            best_acc_test,
            mode=False,
            temp_test=temp_test,
        )
        if mem_stats_active:
            _write_mem_stats(train_writer, train_mem_stats.summary(), 'train', epoch, mem_stats_dir)
            _write_mem_stats(test_writer, test_mem_stats.summary(), 'test', epoch, mem_stats_dir)
        if is_save:
            torch.save(unwrap_model(model).state_dict(), f'./experiments/param/{model_name}/best_test_mpjpe.pth')

    train_writer.close()
    test_writer.close()
    print('model_name: ', model_name)
