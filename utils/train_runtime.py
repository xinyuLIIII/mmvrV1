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


def maybe_compile_model(model, compile_enabled):
    if not compile_enabled:
        return model
    if not hasattr(torch, 'compile'):
        raise RuntimeError('torch.compile is unavailable in the current PyTorch build.')
    return torch.compile(model)


def unwrap_model(model):
    return getattr(model, '_orig_mod', model)
