"""Resumable driver for the 3-fold LOSO pilot (A gesture-only / B dual / C dual+bigrl).

Every (model, fold, seed) run calls train_multitask_disent.py --loso_person ...
and writes experiments/loso_pilot/runs/<run_id>.json when it finishes.  A run
whose JSON already exists is skipped, so the driver can be killed and restarted.
results.csv is rebuilt from all finished JSONs after each completion.

  python scripts/run_loso_pilot.py --jobs 4 --epochs 60 [--device auto|cuda|cpu]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'experiments', 'loso_pilot')
RUN_DIR = os.path.join(OUT_DIR, 'runs')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
CSV_PATH = os.path.join(OUT_DIR, 'results.csv')

# 01 / 07 / 11 = 8 名保留者排序后的第 0 / 3 / 7 位，覆盖 person_id 低、中、高段
FOLDS = ('01', '07', '11')
SEEDS = (0, 1, 2)
MODELS = {
    'A': ['--task', 'gesture', '--disent_method', 'none'],
    'B': ['--task', 'dual', '--disent_method', 'none'],
    'C': ['--task', 'dual', '--disent_method', 'bigrl'],
}
CSV_FIELDS = [
    'model', 'task', 'method', 'fold_person', 'seed',
    'gesture_acc_heldout', 'identity_acc_val', 'selected_epoch', 'val_gesture_acc',
    'gesture_acc_heldout_final', 'identity_acc_val_final', 'val_gesture_acc_final',
    'epochs', 'train_size', 'val_size', 'heldout_size', 'wall_sec',
]

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + msg, flush=True)


def run_id(model, fold, seed):
    return f'{model}_p{fold}_s{seed}'


def common_args(a):
    # Identical for A/B/C.  C additionally gets the adversarial schedule below,
    # which only has an effect when --disent_method != none.
    return [
        '--dataset_root_kpt', a.dataset_root_kpt,
        '--device', a.device,
        '--epoch', str(a.epochs),
        '--batch_size', '32',
        '--lr', '1e-4',
        '--weight_decay', '1e-4',
        '--lr_drop', '200',
        '--clip_max_norm', '0.1',
        '--identity_loss_weight', '1.0',
        '--num_workers', '0',
        '--no_pin_memory',
        '--disent_lambda_max', '0.3',
        '--disent_adv_start_epoch', str(a.adv_start),
        '--disent_adv_ramp_epochs', str(a.adv_ramp),
    ]


def load_result(path):
    try:
        with open(path) as fh:
            res = json.load(fh)
    except (OSError, ValueError):
        return None
    return res if res.get('selected') and res.get('final') else None


def rebuild_csv():
    rows = []
    for model in MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                rid = run_id(model, fold, seed)
                res = load_result(os.path.join(RUN_DIR, rid + '.json'))
                if res is None:
                    continue
                sel, fin = res['selected'], res['final']
                rows.append({
                    'model': model, 'task': res['task'], 'method': res['method'],
                    'fold_person': fold, 'seed': seed,
                    'gesture_acc_heldout': round(sel['heldout_gesture_acc'], 3),
                    'identity_acc_val': round(sel['val_identity_acc'], 3),
                    'selected_epoch': sel['epoch'],
                    'val_gesture_acc': round(sel['val_gesture_acc'], 3),
                    'gesture_acc_heldout_final': round(fin['heldout_gesture_acc'], 3),
                    'identity_acc_val_final': round(fin['val_identity_acc'], 3),
                    'val_gesture_acc_final': round(fin['val_gesture_acc'], 3),
                    'epochs': res['epochs'], 'train_size': res['train_size'],
                    'val_size': res['val_size'], 'heldout_size': res['heldout_size'],
                    'wall_sec': res.get('wall_sec', ''),
                })
    tmp = CSV_PATH + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, CSV_PATH)
    return len(rows)


def run_one(a, model, fold, seed):
    rid = run_id(model, fold, seed)
    out_json = os.path.join(RUN_DIR, rid + '.json')
    log_path = os.path.join(LOG_DIR, rid + '.log')
    cmd = [sys.executable, '-u', os.path.join(ROOT, 'train_multitask_disent.py'),
           *MODELS[model], *common_args(a),
           '--seed', str(seed), '--loso_person', fold,
           '--loso_out', out_json + '.partial',
           '--disent_run_suffix', f'_loso_p{fold}_s{seed}']
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = str(a.threads)
    env['TQDM_DISABLE'] = '1'
    log(f'start {rid} (threads={a.threads})')
    t0 = time.time()
    with open(log_path, 'w') as fh:
        fh.write(' '.join(cmd) + '\n\n')
        fh.flush()
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    partial = out_json + '.partial'
    if rc != 0 or load_result(partial) is None:
        log(f'FAIL {rid} rc={rc} after {wall:.0f}s, see {log_path}')
        return rid, False
    with open(partial) as fh:
        res = json.load(fh)
    res['wall_sec'] = round(wall, 1)
    with open(partial, 'w') as fh:
        json.dump(res, fh, indent=1)
    os.replace(partial, out_json)
    with _lock:
        n = rebuild_csv()
    log(f'done {rid} in {wall:.0f}s  heldout_g={res["selected"]["heldout_gesture_acc"]:.2f} '
        f'val_id={res["selected"]["val_identity_acc"]:.2f}  ({n} rows in results.csv)')
    return rid, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--adv_start', type=int, default=None,
                    help='C only; default keeps the original 40-of-60 ratio: round(epochs*2/3)')
    ap.add_argument('--adv_ramp', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=None, help='torch threads per job')
    ap.add_argument('--dataset_root_kpt', default='./data/mmVR')
    ap.add_argument('--only', default='', help='comma list of run ids to restrict to')
    ap.add_argument('--device', default='auto',
                    help='auto (cuda if available, else cpu) | cuda | cuda:N | cpu')
    a = ap.parse_args()
    if a.device == 'auto':
        import torch
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Original Route A schedule: adversary starts at ep 40, ramps over 40 (60-ep runs).
    if a.adv_start is None:
        a.adv_start = round(a.epochs * 40 / 60)
    if a.adv_ramp is None:
        a.adv_ramp = round(a.epochs * 40 / 60)
    if a.threads is None:
        a.threads = max(1, (os.cpu_count() or 4) // a.jobs)

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    rebuild_csv()

    # seed-major order: each completed "seed sweep" covers every (model, fold)
    todo = []
    for seed in SEEDS:
        for fold in FOLDS:
            for model in MODELS:
                rid = run_id(model, fold, seed)
                if a.only and rid not in a.only.split(','):
                    continue
                if load_result(os.path.join(RUN_DIR, rid + '.json')) is not None:
                    log(f'skip {rid} (already finished)')
                    continue
                todo.append((model, fold, seed))
    log(f'driver pid={os.getpid()} device={a.device} epochs={a.epochs} adv_start={a.adv_start} '
        f'adv_ramp={a.adv_ramp} jobs={a.jobs} threads/job={a.threads} todo={len(todo)}')

    failed = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = [pool.submit(run_one, a, *t) for t in todo]
        for fut in as_completed(futs):
            rid, ok = fut.result()
            if not ok:
                failed.append(rid)
    log(f'all done; failed={failed}')


if __name__ == '__main__':
    main()
