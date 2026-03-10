# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import logging
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
import torch
import torch.distributed as dist
from torch import Tensor
import matplotlib.pyplot as plt
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
import pandas as pd

if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class MemStatsAccumulator(object):
    def __init__(self, eps=1e-6, sparsity_thresh=1e-3):
        self.eps = eps
        self.sparsity_thresh = sparsity_thresh
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = None
        self.max = None
        self.abs_sum = 0.0
        self.l1 = 0.0
        self.l2_sq = 0.0
        self.sparse_count = 0

    def update(self, tensor):
        x = tensor.detach().float().view(-1)
        if x.numel() == 0:
            return
        self.count += x.numel()
        self.sum += x.sum().item()
        self.sum_sq += (x * x).sum().item()
        self.abs_sum += x.abs().sum().item()
        self.l1 += x.abs().sum().item()
        self.l2_sq += (x * x).sum().item()
        self.sparse_count += (x.abs() < self.sparsity_thresh).sum().item()
        cur_min = x.min().item()
        cur_max = x.max().item()
        self.min = cur_min if self.min is None else min(self.min, cur_min)
        self.max = cur_max if self.max is None else max(self.max, cur_max)

    def summary(self):
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "l1": 0.0,
                "l2": 0.0,
                "abs_mean": 0.0,
                "sparsity": 0.0,
            }
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        std = var ** 0.5
        l2 = self.l2_sq ** 0.5
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "min": self.min,
            "max": self.max,
            "l1": self.l1,
            "l2": l2,
            "abs_mean": self.abs_sum / self.count,
            "sparsity": float(self.sparse_count) / float(self.count),
        }


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def collate_fn_kpt(batch):
    batch = list(zip(*batch))
    # batch[0] = torch.stack(batch[0], dim=0)
    # batch[1] = torch.stack(batch[1], dim=0)
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0], dim=0)
    batch[1] = torch.stack(batch[1], dim=0)
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split(".")[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


POSE_METRIC_PRINT_ORDER = (
    'MPJPE',
    'MPJDLE',
    'MPJDLE_h',
    'MPJDLE_v',
    'MPJDLE_d',
    'mpjpe_thumb',
    'mpjdle_h_thumb',
    'mpjdle_v_thumb',
    'mpjdle_d_thumb',
    'mpjpe_index',
    'mpjdle_h_index',
    'mpjdle_v_index',
    'mpjdle_d_index',
    'mpjpe_middle',
    'mpjdle_h_middle',
    'mpjdle_v_middle',
    'mpjdle_d_middle',
    'mpjpe_ring',
    'mpjdle_h_ring',
    'mpjdle_v_ring',
    'mpjdle_d_ring',
    'mpjpe_pinky',
    'mpjdle_h_pinky',
    'mpjdle_v_pinky',
    'mpjdle_d_pinky',
)


def _print_pose_metrics(metric_summary):
    for name in POSE_METRIC_PRINT_ORDER:
        print(f'{name}:', round(metric_summary.get(name, 0.0), 4))


def log_save(model_name, writer, logging, epoch, loss_summary, metric_summary, criterion, best_mpjpe, best_acc, mode,
             temp_test=0.0):
    print("\nepoch:", epoch + 1)
    split = 'train' if mode else 'test'
    split_title = 'train' if mode else 'test'
    current_mpjpe = round(metric_summary.get('MPJPE', 0.0), 4)
    current_accuracy = round(metric_summary.get('accuracy', 0.0), 4)
    best_acc = max(best_acc, current_accuracy)
    is_save = current_mpjpe < best_mpjpe
    best_mpjpe = min(best_mpjpe, current_mpjpe)

    print(f"------------------------------{split_title}----------------------------------------")
    print("loss:", round(loss_summary.get('loss', 0.0), 4))
    print("loss['kpt']:", round(loss_summary.get('loss_dict', {}).get('loss_kpt', 0.0), 4))
    print("loss['cls']:", round(loss_summary.get('loss_dict', {}).get('loss_ce', 0.0), 4))
    print("class_error:", round(loss_summary.get('loss_dict', {}).get('class_error', 0.0), 4))
    runtime_summary = loss_summary.get('runtime', {})
    if runtime_summary:
        print("data_time:", round(runtime_summary.get('data_time', 0.0), 4))
        print("step_time:", round(runtime_summary.get('step_time', 0.0), 4))
        print("samples_per_sec:", round(runtime_summary.get('samples_per_sec', 0.0), 4))
    print(f"{'Training' if mode else 'Test'} accuracy:", current_accuracy)
    _print_pose_metrics(metric_summary)
    print("best MPJPE:", best_mpjpe, " and acc:", best_acc)

    params = {
        'epoch': epoch + 1,
        'loss': round(loss_summary.get('loss', 0.0), 4),
        'loss_kpt': round(loss_summary.get('loss_dict', {}).get('loss_kpt', 0.0), 4),
        'loss_cls': round(loss_summary.get('loss_dict', {}).get('loss_ce', 0.0), 4),
        'class_error': round(loss_summary.get('loss_dict', {}).get('class_error', 0.0), 4),
        'data_time': round(runtime_summary.get('data_time', 0.0), 4),
        'step_time': round(runtime_summary.get('step_time', 0.0), 4),
        'samples_per_sec': round(runtime_summary.get('samples_per_sec', 0.0), 4),
        'Training_accuracy': current_accuracy,
        'best_MPJPE': best_mpjpe,
        'best_acc': best_acc,
    }
    for name in POSE_METRIC_PRINT_ORDER:
        params[name] = round(metric_summary.get(name, 0.0), 4)

    logging.info(params)
    writer.add_scalar(f'Loss/{split}_total_loss', params['loss'], epoch + 1)
    writer.add_scalar(f'Loss/{split}_loss_kpt', params['loss_kpt'], epoch + 1)
    writer.add_scalar(f'Loss/{split}_loss_cls', params['loss_cls'], epoch + 1)
    writer.add_scalar(f'Runtime/{split}_data_time', params['data_time'], epoch + 1)
    writer.add_scalar(f'Runtime/{split}_step_time', params['step_time'], epoch + 1)
    writer.add_scalar(f'Runtime/{split}_samples_per_sec', params['samples_per_sec'], epoch + 1)
    writer.add_scalar('Accuracy/class_error', params['class_error'], epoch + 1)
    writer.add_scalar(f'Accuracy/{split}_accuracy', params['Training_accuracy'], epoch + 1)
    writer.add_scalar(f'Accuracy/{split}_MPJPE', params['MPJPE'], epoch + 1)
    writer.add_scalar(f'Accuracy/{split}_best_MPJPE', params['best_MPJPE'], epoch + 1)
    writer.add_scalar(f'Accuracy/{split}_best_acc', params['best_acc'], epoch + 1)

    if not mode and current_accuracy > temp_test:
        conf_matrix = criterion.conf_matrix
        plt.matshow(conf_matrix, cmap=plt.cm.Reds)
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix)):
                plt.text(j, i, str(conf_matrix[i][j]), horizontalalignment='center', verticalalignment='center',
                         fontsize=4)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(
            './experiments/conf_matrix/' + model_name + '/' + model_name + '_' +
            str(epoch) + '_' + str(params['Training_accuracy']) + '.jpg', dpi=300)
        criterion.write_to_file(
            conf_matrix,
            './experiments/weights/' + model_name + '/' + model_name + '_' +
            str(epoch) + '_' + str(params['Training_accuracy'])
        )
        plt.close()
        temp_test = current_accuracy
    return best_mpjpe, best_acc, temp_test, is_save


def log_save_kpt(model_name, writer, logging, epoch, losses, size, best_train_acc, train_acc, conf_matrix, mode):
    print("\nepoch:", epoch + 1)
    is_save = False
    if mode:
        best_train_acc = max(train_acc, best_train_acc)
        print("------------------------------train----------------------------------------")
        print("loss:", losses / size)
        print("Train accuracy:", train_acc)
        print("max_train: ", best_train_acc)

        params = {
            'epoch': epoch + 1,
            'loss': round(losses / size, 4),
            'Training_accuracy': train_acc,
            'max_train': best_train_acc
        }
        logging.info(params)
        writer.add_scalar('Log/train_loss', params['loss'], epoch + 1)
        writer.add_scalar('Log/train_accuracy', params['Training_accuracy'], epoch + 1)
        writer.add_scalar('Log/max_train', params['max_train'], epoch + 1)
    else:
        if train_acc > best_train_acc:
            is_save = True
        best_train_acc = max(train_acc, best_train_acc)
        print("------------------------------test----------------------------------------")
        print("loss:", losses / size)
        print("Test accuracy:", train_acc)
        print("max_test: ", best_train_acc)

        params = {
            'epoch': epoch + 1,
            'loss': round(losses / size, 4),
            'Testing_accuracy': train_acc,
            'max_test': best_train_acc
        }
        logging.info(params)
        writer.add_scalar('Log/test_loss', params['loss'], epoch + 1)
        writer.add_scalar('Log/test_accuracy', params['Testing_accuracy'], epoch + 1)
        writer.add_scalar('Log/max_test', params['max_test'], epoch + 1)
        # params_list.append(params)
        #     temp_test = 0
        if is_save:
            plt.matshow(conf_matrix, cmap=plt.cm.Reds)
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix)):
                    plt.text(j, i, str(conf_matrix[i][j]), horizontalalignment='center', verticalalignment='center',
                             fontsize=4)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # plt.show()

            plt.savefig(
                './experiments/conf_matrix/' + model_name + '/' + model_name + '_' +
                str(epoch) + '_' + str(params['Testing_accuracy']) + '.jpg', dpi=300)
            write_to_file(conf_matrix, './experiments/weights/' + model_name + '/' +
                                    model_name + '_' + str(epoch) + '_' + str(params['Testing_accuracy']))
            plt.close()
    return best_train_acc, is_save


def get_conf_matrix(pred, truth, conf_matrix):
    p = pred.tolist()
    l = truth.tolist()
    for i in range(len(p)):
        conf_matrix[l[i]][p[i]] += 1
    return conf_matrix

def write_to_file(conf_matrix, path):
    conf_matrix_m = []
    for row in conf_matrix:
        base = sum(row)
        if base == 0:
            conf_matrix_m.append([format(0, '.2f') for _ in row])
            continue
        conf_matrix_m.append([format(value / base, '.2f') for value in row])
    df = pd.DataFrame(conf_matrix_m)
    df.to_csv(path + '.csv')

def create_log(filename):
    train_logger = logging.getLogger('training')
    train_logger.setLevel(logging.INFO)
    for handler in list(train_logger.handlers):
        train_logger.removeHandler(handler)
        handler.close()

    train_handler = logging.FileHandler(filename + '_train.log')
    train_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    train_handler.setFormatter(train_formatter)

    train_logger.addHandler(train_handler)

    test_logger = logging.getLogger('testing')
    test_logger.setLevel(logging.INFO)
    for handler in list(test_logger.handlers):
        test_logger.removeHandler(handler)
        handler.close()

    test_handler = logging.FileHandler(filename + '_test.log')
    test_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    test_handler.setFormatter(test_formatter)

    test_logger.addHandler(test_handler)

    return train_logger, test_logger


def criterion_init(criterion):
    if hasattr(criterion, 'reset_epoch_stats'):
        criterion.reset_epoch_stats()
        return

    criterion.correct_num = 0
