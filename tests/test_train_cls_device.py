import pathlib
import sys
import importlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_cls.py"])
    train_cls = importlib.import_module("train_cls")
    monkeypatch.setattr(train_cls.torch.cuda, 'is_available', lambda: False)
    device = train_cls.resolve_device('cuda')
    assert device.type == 'cpu'
