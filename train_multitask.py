import os
import random

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset.dataset_identity import build_dataset, NUM_GESTURE, NUM_IDENTITY
from models.multitask_resnet import build_multitask_model
from utils import misc
from utils.train_runtime import (
    autocast_context, build_dataloader_kwargs, configure_runtime,
    create_grad_scaler, ensure_experiment_dirs, get_amp_settings,
    maybe_compile_model, resolve_device as runtime_resolve_device, unwrap_model,
)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

TASK_KEYS = {'gesture': ['gesture'], 'identity': ['identity'],
             'dual': ['gesture', 'identity']}


def resolve_model_name(task):
    return f'train_mt_{task}'


def compute_loss(outputs, g_label, id_label, task, id_weight, loss_fn):
    loss = None
    if outputs['gesture'] is not None:
        loss = loss_fn(outputs['gesture'], g_label)
    if outputs['identity'] is not None:
        id_ce = loss_fn(outputs['identity'], id_label)
        if loss is None:
            loss = id_ce                      # identity-only: raw CE, no weight
        else:
            loss = loss + id_weight * id_ce   # dual: weighted sum
    return loss


def update_metrics(outputs, g_label, id_label, task, correct, total):
    if outputs['gesture'] is not None:
        pred = outputs['gesture'].argmax(dim=1)
        correct['gesture'] += int(pred.eq(g_label).sum().item())
        total['gesture'] += int(g_label.shape[0])
    if outputs['identity'] is not None:
        pred = outputs['identity'].argmax(dim=1)
        correct['identity'] += int(pred.eq(id_label).sum().item())
        total['identity'] += int(id_label.shape[0])


def resolve_device(requested_device):
    try:
        return runtime_resolve_device(requested_device)
    except RuntimeError:
        device = torch.device(requested_device)
        if device.type == 'cuda':
            print('CUDA requested but unavailable. Falling back to CPU.')
            return torch.device('cpu')
        raise


from contextlib import nullcontext


def run_epoch(model, loader, device, loss_fn, amp_device_type, amp_enabled,
              task, id_weight, optimizer=None, scaler=None):
    is_train = optimizer is not None
    if is_train and scaler is None:
        raise ValueError('GradScaler is required for training epochs.')
    loss_sum = 0.0
    correct = {'gesture': 0, 'identity': 0}
    total = {'gesture': 0, 'identity': 0}
    model.train() if is_train else model.eval()
    grad_context = nullcontext() if is_train else torch.no_grad()

    for kpt, g_label, id_label in tqdm(loader):
        kpt = kpt.to(device, non_blocking=True)
        g_label = g_label.to(device, non_blocking=True)
        id_label = id_label.to(device, non_blocking=True)
        batch_size = int(g_label.shape[0])
        with grad_context:
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_device_type, enabled=amp_enabled):
                outputs = model(kpt)
                loss = compute_loss(outputs, g_label, id_label, task, id_weight, loss_fn)
            update_metrics(outputs, g_label, id_label, task, correct, total)
            loss_sum += float(loss.detach().item()) * batch_size
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    return loss_sum, correct, total


def _acc(correct, total, key):
    return 100.0 * correct[key] / total[key] if total[key] else 0.0


def main():
    args = config.args

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_path = './experiments/'
    task = args.task
    model_name = resolve_model_name(task)
    print('model_name:', model_name)

    device = resolve_device(args.device)
    configure_runtime(device, args.matmul_precision)
    amp_enabled, amp_device_type = get_amp_settings(device, args.amp)
    ensure_experiment_dirs(model_name, base_dir=save_path)

    train_ds = build_dataset(args.dataset_root_kpt, True,
                             test_size=args.test_size, seed=args.seed)
    test_ds = build_dataset(args.dataset_root_kpt, False,
                            test_size=args.test_size, seed=args.seed)
    train_kwargs = build_dataloader_kwargs(
        args.num_workers, args.pin_memory, shuffle=True, drop_last=True,
        persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)
    test_kwargs = build_dataloader_kwargs(
        args.num_workers, args.pin_memory, shuffle=False, drop_last=False,
        persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, **train_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, **test_kwargs)
    print('train_size:', len(train_ds), 'test_size:', len(test_ds))

    model = build_multitask_model(task, NUM_GESTURE, NUM_IDENTITY).to(device)
    loss_ce = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.5)
    model = maybe_compile_model(model, args.compile)

    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    scaler = create_grad_scaler(amp_device_type, enabled=amp_enabled)
    keys = TASK_KEYS[task]
    best_score = 0.0

    for i in range(args.epoch):
        print('epoch:', i)
        tr_loss, tr_correct, tr_total = run_epoch(
            model, train_loader, device, loss_ce, amp_device_type, amp_enabled,
            task, args.identity_loss_weight, optimizer=optimizer, scaler=scaler)
        lr_scheduler.step()
        te_loss, te_correct, te_total = run_epoch(
            model, test_loader, device, loss_ce, amp_device_type, amp_enabled,
            task, args.identity_loss_weight)

        msg = {'epoch': i}
        for k in keys:
            msg[f'train_{k}_acc'] = round(_acc(tr_correct, tr_total, k), 3)
            msg[f'test_{k}_acc'] = round(_acc(te_correct, te_total, k), 3)
            train_writer.add_scalar(f'{k}_acc', _acc(tr_correct, tr_total, k), i)
            test_writer.add_scalar(f'{k}_acc', _acc(te_correct, te_total, k), i)
        train_writer.add_scalar('loss', tr_loss / max(len(train_ds), 1), i)
        test_writer.add_scalar('loss', te_loss / max(len(test_ds), 1), i)
        print(msg)
        train_logger.info(msg)
        test_logger.info(msg)

        score = sum(_acc(te_correct, te_total, k) for k in keys) / len(keys)
        if score > best_score:
            best_score = score
            torch.save(unwrap_model(model).state_dict(),
                       f'./experiments/param/{model_name}/best_test_mpjpe.pth')

    train_writer.close()
    test_writer.close()
    print('done:', model_name, 'best_score:', round(best_score, 3))


if __name__ == '__main__':
    main()

