import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset.dataset_kpt import build_dataset
from models.ResNet import resnet50
from utils import misc
from utils.train_runtime import (
    autocast_context,
    build_dataloader_kwargs,
    configure_runtime,
    create_grad_scaler,
    ensure_experiment_dirs,
    get_amp_settings,
    maybe_compile_model,
    resolve_device as runtime_resolve_device,
    unwrap_model,
)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def resolve_device(requested_device: str) -> torch.device:
    try:
        return runtime_resolve_device(requested_device)
    except RuntimeError:
        device = torch.device(requested_device)
        if device.type == 'cuda':
            print('CUDA requested but unavailable. Falling back to CPU.')
            return torch.device('cpu')
        raise


def build_classification_dataloaders(args):
    train_dataset = build_dataset(args.dataset_root_kpt, True)
    test_dataset = build_dataset(args.dataset_root_kpt, False)

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
        **train_loader_kwargs,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        **test_loader_kwargs,
    )
    return train_dataset, test_dataset, train_loader, test_loader, train_loader_kwargs, test_loader_kwargs


def build_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.5)
    return optimizer, scheduler


def create_run_start_record(
    model_name,
    args,
    device,
    train_size,
    test_size,
    train_loader_kwargs,
    test_loader_kwargs,
    optimizer,
    scheduler,
    amp_enabled,
):
    return {
        'run_event': 'start',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'pid': os.getpid(),
        'model_name': model_name,
        'argv': sys.argv,
        'device': str(device),
        'dataset_root_kpt': args.dataset_root_kpt,
        'train_size': train_size,
        'test_size': test_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'persistent_workers': args.persistent_workers,
        'prefetch_factor': args.prefetch_factor if args.num_workers > 0 else None,
        'train_loader': train_loader_kwargs,
        'test_loader': test_loader_kwargs,
        'optimizer': optimizer.__class__.__name__,
        'lr': float(optimizer.param_groups[0]['lr']),
        'weight_decay': float(optimizer.param_groups[0]['weight_decay']),
        'scheduler': scheduler.__class__.__name__,
        'lr_drop': getattr(scheduler, 'step_size', None),
        'amp_enabled': amp_enabled,
        'compile_enabled': args.compile,
        'matmul_precision': args.matmul_precision,
    }


def log_run_start(train_logger, test_logger, run_record):
    train_logger.info(run_record)
    test_logger.info(run_record)
    print('run_start:', json.dumps(run_record, sort_keys=True))


def run_classification_epoch(
    model,
    data_loader,
    device,
    loss_fn,
    amp_device_type,
    amp_enabled,
    num_class,
    optimizer=None,
    scaler=None,
):
    is_train = optimizer is not None
    if is_train and scaler is None:
        raise ValueError('GradScaler is required for training epochs.')

    loss_sum = 0.0
    correct = 0
    sample_count = 0
    conf_matrix = [[0 for _ in range(num_class)] for _ in range(num_class)]

    if is_train:
        model.train()
    else:
        model.eval()

    grad_context = nullcontext() if is_train else torch.no_grad()
    for kpt, label in tqdm(data_loader):
        kpt = kpt.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        batch_size = int(label.shape[0])
        sample_count += batch_size

        with grad_context:
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_device_type, enabled=amp_enabled):
                output = model(kpt)
                loss = loss_fn(output, label)
            prediction = output.argmax(dim=1)
            correct += int(prediction.eq(label).sum().item())
            loss_sum += float(loss.detach().item()) * batch_size
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                conf_matrix = misc.get_conf_matrix(prediction, label, conf_matrix)

    if sample_count == 0:
        raise ValueError('Classification epoch produced zero samples.')

    return loss_sum, correct, sample_count, conf_matrix


def main():
    args = config.args
    save_path = './experiments/'
    model_name = 'train'
    num_class = 8

    print(model_name)
    device = resolve_device(args.device)
    configure_runtime(device, args.matmul_precision)
    amp_enabled, amp_device_type = get_amp_settings(device, args.amp)
    if model_name != 'test':
        ensure_experiment_dirs(model_name, base_dir=save_path)

    _, _, train_loader, test_loader, train_loader_kwargs, test_loader_kwargs = build_classification_dataloaders(args)
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    print('model_name: ', model_name)
    print('train_size: ', train_size)
    print('test_size: ', test_size)

    model = resnet50(num_classes=num_class).to(device)
    loss_ce = nn.CrossEntropyLoss().to(device)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(model, args)
    model = maybe_compile_model(model, args.compile)

    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    scaler = create_grad_scaler(amp_device_type, enabled=amp_enabled)
    run_record = create_run_start_record(
        model_name,
        args,
        device,
        train_size,
        test_size,
        train_loader_kwargs,
        test_loader_kwargs,
        optimizer,
        lr_scheduler,
        amp_enabled,
    )
    log_run_start(train_logger, test_logger, run_record)

    best_acc_train = 0.0
    best_acc_test = 0.0

    for i in range(args.epoch):
        print('epoch: ', i)
        train_losses, correct_train, train_sample_count, train_conf_matrix = run_classification_epoch(
            model,
            train_loader,
            device,
            loss_ce,
            amp_device_type,
            amp_enabled,
            num_class,
            optimizer=optimizer,
            scaler=scaler,
        )
        lr_scheduler.step()
        train_acc = 100 * float(correct_train) / train_sample_count
        best_acc_train, _ = misc.log_save_kpt(
            model_name,
            train_writer,
            train_logger,
            i,
            train_losses,
            train_sample_count,
            best_acc_train,
            train_acc,
            train_conf_matrix,
            mode=True,
        )

        test_losses, correct_test, test_sample_count, conf_matrix = run_classification_epoch(
            model,
            test_loader,
            device,
            loss_ce,
            amp_device_type,
            amp_enabled,
            num_class,
        )
        test_acc = 100 * float(correct_test) / test_sample_count
        best_acc_test, is_save = misc.log_save_kpt(
            model_name,
            test_writer,
            test_logger,
            i,
            test_losses,
            test_sample_count,
            best_acc_test,
            test_acc,
            conf_matrix,
            mode=False,
        )
        if is_save:
            torch.save(unwrap_model(model).state_dict(), f'./experiments/param/{model_name}/best_test_mpjpe.pth')

    train_writer.close()
    test_writer.close()
    print('model_name: ', model_name)


if __name__ == '__main__':
    main()

