from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def test_task_arg_exists_with_default_dual():
    assert hasattr(config.args, 'task')
    assert config.args.task == 'dual'


def test_identity_loss_weight_default_is_one():
    assert hasattr(config.args, 'identity_loss_weight')
    assert config.args.identity_loss_weight == 1.0
