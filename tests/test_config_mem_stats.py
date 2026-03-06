import importlib
import sys


def test_config_mem_stats_defaults():
    sys.argv = ["test"]
    config = importlib.import_module("config")
    importlib.reload(config)
    assert hasattr(config.args, "mem_stats")
    assert config.args.mem_stats is False
