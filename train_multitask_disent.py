"""Route A entry point: bounded bidirectional GRL on private subspaces.

Additive file.  train_multitask.py / train_cls.py / dataset_kpt.py / ResNet.py
are untouched.  Until config.py gets the 2-line edit described in the design,
this file registers the new flags on config.parser itself, so it runs today.

  python train_multitask_disent.py --dataset_root_kpt ./data/mmVR \
      --task dual --disent_method bigrl --disent_lambda_max 0.3
"""

import json
import os
import random
from contextlib import nullcontext

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset.dataset_identity import (
    build_dataset, build_loso_datasets, NUM_GESTURE, NUM_IDENTITY,
)
from models.disentangle import (
    LambdaController, add_disent_args, config_from_args, floor_gate,
    scheduled_lambda,
)
from models.multitask_resnet_disent import build_disent_model
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


def get_args():
    """Shim: once config.py carries add_disent_args(parser), just use config.args."""
    if hasattr(config.args, 'disent_method'):
        return config.args
    add_disent_args(config.parser)
    parsed, _ = config.parser.parse_known_args()
    return parsed


def resolve_device(requested_device):
    try:
        return runtime_resolve_device(requested_device)
    except RuntimeError:
        device = torch.device(requested_device)
        if device.type == 'cuda':
            print('CUDA requested but unavailable. Falling back to CPU.')
            return torch.device('cpu')
        raise


def run_disent_epoch(model, data_loader, device, task, ce_loss,
                     identity_loss_weight, amp_device_type, amp_enabled,
                     optimizer=None, scaler=None,
                     cfg=None, recon_weight=0.0, clip_max_norm=0.0):
    """Same first-6 positional contract as run_multitask_epoch."""
    is_train = optimizer is not None
    use_disc = cfg is not None and cfg.enabled and task == 'dual'

    acc = {'loss_sum': 0.0, 'loss_g': 0.0, 'loss_id': 0.0,
           'loss_adv_id': 0.0, 'loss_adv_g': 0.0, 'loss_rec': 0.0}
    g_correct = 0
    id_correct = 0
    adv_id_correct = 0
    adv_g_correct = 0
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
                loss_g = loss_id = loss_ai = loss_ag = loss_rec = None

                if task in ('gesture', 'dual'):
                    loss_g = ce_loss(out['gesture'], g_label)
                    loss = loss + loss_g
                if task in ('identity', 'dual'):
                    loss_id = ce_loss(out['identity'], id_label)
                    loss = loss + identity_loss_weight * loss_id

                # DANN: the discriminator's own CE keeps coefficient 1.0.
                # Adversarial strength lives entirely inside the GRL lambda,
                # so the discriminator always trains at full speed.
                if use_disc and out['adv_identity'] is not None:
                    loss_ai = ce_loss(out['adv_identity'], id_label)
                    loss = loss + loss_ai
                if use_disc and out['adv_gesture'] is not None:
                    loss_ag = ce_loss(out['adv_gesture'], g_label)
                    loss = loss + loss_ag
                if out['recon'] is not None and recon_weight > 0:
                    loss_rec = F.mse_loss(out['recon'], out['recon_target'])
                    loss = loss + recon_weight * loss_rec

            if out['gesture'] is not None:
                g_correct += int(out['gesture'].argmax(1).eq(g_label).sum().item())
            if out['identity'] is not None:
                id_correct += int(out['identity'].argmax(1).eq(id_label).sum().item())
            if out['adv_identity'] is not None:
                adv_id_correct += int(out['adv_identity'].argmax(1).eq(id_label).sum().item())
            if out['adv_gesture'] is not None:
                adv_g_correct += int(out['adv_gesture'].argmax(1).eq(g_label).sum().item())

            acc['loss_sum'] += float(loss.detach().item()) * bs
            for key, val in (('loss_g', loss_g), ('loss_id', loss_id),
                             ('loss_adv_id', loss_ai), ('loss_adv_g', loss_ag),
                             ('loss_rec', loss_rec)):
                if val is not None:
                    acc[key] += float(val.detach().item()) * bs

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    if clip_max_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                    optimizer.step()

    if count == 0:
        raise ValueError('Multitask epoch produced zero samples.')
    stats = {k: v for k, v in acc.items()}
    stats.update({'g_correct': g_correct, 'id_correct': id_correct,
                  'adv_id_correct': adv_id_correct, 'adv_g_correct': adv_g_correct,
                  'count': count})
    return stats


def main():
    args = get_args()
    cfg = config_from_args(args)
    task = args.task
    save_path = './experiments/'
    tag = cfg.method if cfg.enabled else 'base'
    loso = args.loso_person is not None
    model_name = f'train_mt_{task}_{tag}{args.disent_run_suffix}'

    print('model_name:', model_name)
    torch.manual_seed(args.seed)
    if loso:
        random.seed(args.seed)
        np.random.seed(args.seed)
    device = resolve_device(args.device)
    configure_runtime(device, args.matmul_precision)
    amp_enabled, amp_device_type = get_amp_settings(device, args.amp)
    ensure_experiment_dirs(model_name, base_dir=save_path)

    if loso:
        # test_ds = 见过的人的验证块：选模型 / LambdaController 只看它
        train_ds, test_ds, heldout_ds = build_loso_datasets(
            args.dataset_root_kpt, args.loso_person)
    else:
        train_ds = build_dataset(args.dataset_root_kpt, True)
        test_ds = build_dataset(args.dataset_root_kpt, False)
        heldout_ds = None
    train_kwargs = build_dataloader_kwargs(
        args.num_workers, args.pin_memory, shuffle=True, drop_last=True,
        persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)
    test_kwargs = build_dataloader_kwargs(
        args.num_workers, args.pin_memory, shuffle=False, drop_last=False,
        persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, **train_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, **test_kwargs)
    print('train_size:', len(train_ds), 'test_size:', len(test_ds))
    heldout_loader = None
    if loso:
        heldout_loader = DataLoader(heldout_ds, batch_size=args.batch_size, **test_kwargs)
        print('loso_person:', args.loso_person, 'heldout_size:', len(heldout_ds))

    model = build_disent_model(task, NUM_GESTURE, NUM_IDENTITY, cfg=cfg).to(device)
    ce = nn.CrossEntropyLoss().to(device)

    disc_params = list(model.discriminator_parameters())
    main_params = list(model.main_parameters())
    groups = [{'params': main_params, 'lr': args.lr}]
    if disc_params:
        # DANN / 2110.12163: keep the discriminator strictly stronger than the
        # feature path, otherwise the reversed gradient is noise.
        groups.append({'params': disc_params, 'lr': args.lr * cfg.disc_lr_mult})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.5)
    model = maybe_compile_model(model, args.compile)

    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    scaler = create_grad_scaler(amp_device_type, enabled=amp_enabled)

    controller = LambdaController(cfg)
    ema_adv_id = 1.0     # discriminator train-acc EMA on f_G (identity)
    ema_adv_g = 1.0      # discriminator train-acc EMA on f_I (gesture)
    best_score = 0.0
    best_constrained = 0.0
    # LOSO: 按验证块手势准确率选 epoch；留出者指标只记录、不参与选择
    history = []
    best_val_g = -1.0
    selected = None

    for epoch in range(args.epoch):
        base_lam = scheduled_lambda(epoch, cfg) * controller.scale
        gate_fg = floor_gate(ema_adv_id, NUM_IDENTITY, cfg.gate_margin, cfg.gate_temp)
        gate_fi = floor_gate(ema_adv_g, NUM_GESTURE, cfg.gate_margin, cfg.gate_temp)
        lam_fg = base_lam * gate_fg if cfg.adv_on_fg else 0.0
        lam_fi = base_lam * gate_fi if cfg.adv_on_fi else 0.0
        unwrap_model(model).set_lambda(lam_fg, lam_fi)
        print(f'epoch: {epoch}  lam_fg={lam_fg:.4f} lam_fi={lam_fi:.4f} '
              f'scale={controller.scale:.3f}')

        tr = run_disent_epoch(model, train_loader, device, task, ce,
                              args.identity_loss_weight, amp_device_type, amp_enabled,
                              optimizer=optimizer, scaler=scaler, cfg=cfg,
                              recon_weight=cfg.recon_weight if cfg.enabled else 0.0,
                              clip_max_norm=args.clip_max_norm)
        scheduler.step()

        if cfg.adv_on_fg:
            ema_adv_id = (cfg.gate_ema * ema_adv_id
                          + (1 - cfg.gate_ema) * tr['adv_id_correct'] / tr['count'])
        if cfg.adv_on_fi:
            ema_adv_g = (cfg.gate_ema * ema_adv_g
                         + (1 - cfg.gate_ema) * tr['adv_g_correct'] / tr['count'])

        unwrap_model(model).set_lambda(0.0, 0.0)   # eval: no reversal needed
        te = run_disent_epoch(model, test_loader, device, task, ce,
                              args.identity_loss_weight, amp_device_type, amp_enabled,
                              cfg=cfg,
                              recon_weight=cfg.recon_weight if cfg.enabled else 0.0)

        g_acc = 100.0 * te['g_correct'] / te['count']
        id_acc = 100.0 * te['id_correct'] / te['count']
        if task == 'gesture':
            score = g_acc
        elif task == 'identity':
            score = id_acc
        else:
            score = (g_acc + id_acc) / 2.0
        controller.step(id_acc)

        ho_g_acc = None
        if loso:
            ho = run_disent_epoch(model, heldout_loader, device, task, ce,
                                  args.identity_loss_weight, amp_device_type, amp_enabled,
                                  cfg=cfg,
                                  recon_weight=cfg.recon_weight if cfg.enabled else 0.0)
            ho_g_acc = 100.0 * ho['g_correct'] / ho['count']

        msg = {'epoch': epoch + 1, 'task': task, 'method': tag,
               'train_loss': round(tr['loss_sum'] / tr['count'], 4),
               'test_loss': round(te['loss_sum'] / te['count'], 4),
               'gesture_acc': round(g_acc, 3), 'identity_acc': round(id_acc, 3),
               'score': round(score, 3),
               'lam_fg': round(lam_fg, 4), 'lam_fi': round(lam_fi, 4),
               'probe_id_in_fg_train': round(100.0 * tr['adv_id_correct'] / tr['count'], 2),
               'probe_g_in_fi_train': round(100.0 * tr['adv_g_correct'] / tr['count'], 2),
               'probe_id_in_fg_test': round(100.0 * te['adv_id_correct'] / te['count'], 2),
               'probe_g_in_fi_test': round(100.0 * te['adv_g_correct'] / te['count'], 2),
               'loss_rec': round(tr['loss_rec'] / tr['count'], 5),
               'lam_scale': round(controller.scale, 3)}
        if loso:
            msg['heldout_gesture_acc'] = round(ho_g_acc, 3)
        print(msg)
        test_logger.info(msg)
        train_logger.info({'epoch': epoch + 1,
                           'train_g_acc': round(100.0 * tr['g_correct'] / tr['count'], 3),
                           'train_id_acc': round(100.0 * tr['id_correct'] / tr['count'], 3)})
        for k in ('gesture_acc', 'identity_acc', 'score', 'lam_fg', 'lam_fi',
                  'probe_id_in_fg_test', 'probe_g_in_fi_test'):
            test_writer.add_scalar('mt/' + k, msg[k], epoch + 1)

        if loso:
            row = {'epoch': epoch + 1,
                   'val_gesture_acc': g_acc, 'val_identity_acc': id_acc,
                   'heldout_gesture_acc': ho_g_acc,
                   'train_gesture_acc': 100.0 * tr['g_correct'] / tr['count'],
                   'train_identity_acc': 100.0 * tr['id_correct'] / tr['count'],
                   'lam_fg': lam_fg, 'lam_fi': lam_fi,
                   'probe_id_in_fg_test': msg['probe_id_in_fg_test']}
            history.append(row)
            if task != 'identity' and g_acc > best_val_g:
                best_val_g = g_acc
                selected = row
                if args.loso_save_ckpt:
                    torch.save(unwrap_model(model).state_dict(),
                               f'./experiments/param/{model_name}/best_val_gesture.pth')
            continue

        if score > best_score:
            best_score = score
            torch.save(unwrap_model(model).state_dict(),
                       f'./experiments/param/{model_name}/best.pth')
        # Constrained selection: best gesture subject to identity >= floor.
        if id_acc >= cfg.id_floor and g_acc > best_constrained:
            best_constrained = g_acc
            torch.save(unwrap_model(model).state_dict(),
                       f'./experiments/param/{model_name}/best_constrained.pth')

    train_writer.close()
    test_writer.close()
    if loso:
        result = {'model_name': model_name, 'task': task, 'method': tag,
                  'loso_person': args.loso_person, 'seed': args.seed,
                  'epochs': args.epoch, 'train_size': len(train_ds),
                  'val_size': len(test_ds), 'heldout_size': len(heldout_ds),
                  'selected': selected, 'final': history[-1] if history else None,
                  'history': history}
        if args.loso_out:
            out_dir = os.path.dirname(args.loso_out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.loso_out, 'w') as fh:
                json.dump(result, fh, indent=1)
        print('done:', model_name, 'loso_person:', args.loso_person,
              'selected:', selected)
        return
    print('done:', model_name, 'best_score:', best_score,
          'best_gesture_under_id_floor:', best_constrained)


if __name__ == '__main__':
    main()
