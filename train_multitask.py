import os
import sys
from contextlib import nullcontext

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


def resolve_device(requested_device):
    try:
        return runtime_resolve_device(requested_device)
    except RuntimeError:
        device = torch.device(requested_device)
        if device.type == 'cuda':
            print('CUDA requested but unavailable. Falling back to CPU.')
            return torch.device('cpu')
        raise


def run_multitask_epoch(model, data_loader, device, task, ce_loss,
                        identity_loss_weight, amp_device_type, amp_enabled,
                        optimizer=None, scaler=None):
    is_train = optimizer is not None

    loss_sum = 0.0
    g_correct = 0
    id_correct = 0
    count = 0

    model.train() if is_train else model.eval()
    grad_context = nullcontext() if is_train else torch.no_grad()

    for kpt, g_label, id_label in tqdm(data_loader):
        kpt = kpt.to(device, non_blocking=True)
        g_label = g_label.to(device, non_blocking=True)
        id_label = id_label.to(device, non_blocking=True)
        bs = int(g_label.shape[0])
        count += bs

        with grad_context:
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_device_type, enabled=amp_enabled):
                out = model(kpt)
                loss = torch.zeros((), device=device)
                if task in ('gesture', 'dual'):
                    loss = loss + ce_loss(out['gesture'], g_label)
                if task in ('identity', 'dual'):
                    loss = loss + identity_loss_weight * ce_loss(out['identity'], id_label)
            if out['gesture'] is not None:
                g_correct += int(out['gesture'].argmax(1).eq(g_label).sum().item())
            if out['identity'] is not None:
                id_correct += int(out['identity'].argmax(1).eq(id_label).sum().item())
            loss_sum += float(loss.detach().item()) * bs
            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

    if count == 0:
        raise ValueError('Multitask epoch produced zero samples.')
    return {'loss_sum': loss_sum, 'g_correct': g_correct,
            'id_correct': id_correct, 'count': count}


def main():
    args = config.args
    task = args.task
    save_path = './experiments/'
    model_name = f'train_mt_{task}'

    print('model_name:', model_name)
    device = resolve_device(args.device)
    configure_runtime(device, args.matmul_precision)
    amp_enabled, amp_device_type = get_amp_settings(device, args.amp)
    ensure_experiment_dirs(model_name, base_dir=save_path)

    train_ds = build_dataset(args.dataset_root_kpt, True)
    test_ds = build_dataset(args.dataset_root_kpt, False)
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
    ce = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.5)
    model = maybe_compile_model(model, args.compile)

    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    scaler = create_grad_scaler(amp_device_type, enabled=amp_enabled)

    best_score = 0.0
    for epoch in range(args.epoch):
        print('epoch:', epoch)
        tr = run_multitask_epoch(model, train_loader, device, task, ce,
                                 args.identity_loss_weight, amp_device_type, amp_enabled,
                                 optimizer=optimizer, scaler=scaler)
        scheduler.step()
        te = run_multitask_epoch(model, test_loader, device, task, ce,
                                 args.identity_loss_weight, amp_device_type, amp_enabled)

        g_acc = 100.0 * te['g_correct'] / te['count']
        id_acc = 100.0 * te['id_correct'] / te['count']
        if task == 'gesture':
            score = g_acc
        elif task == 'identity':
            score = id_acc
        else:
            score = (g_acc + id_acc) / 2.0

        msg = {'epoch': epoch + 1, 'task': task,
               'train_loss': round(tr['loss_sum'] / tr['count'], 4),
               'test_loss': round(te['loss_sum'] / te['count'], 4),
               'gesture_acc': round(g_acc, 3), 'identity_acc': round(id_acc, 3),
               'score': round(score, 3)}
        print(msg)
        test_logger.info(msg)
        test_writer.add_scalar('acc/gesture', g_acc, epoch + 1)
        test_writer.add_scalar('acc/identity', id_acc, epoch + 1)
        test_writer.add_scalar('acc/score', score, epoch + 1)

        if score > best_score:
            best_score = score
            torch.save(unwrap_model(model).state_dict(),
                       f'./experiments/param/{model_name}/best.pth')

    train_writer.close()
    test_writer.close()
    print('done:', model_name, 'best_score:', best_score)


if __name__ == '__main__':
    main()
