"""
Test general implementation in haddock3 modules.

Ensures all modules follow the same compatible architecture.
"""
import importlib

import pytest

from haddock.core.exceptions import ConfigurationError
from haddock.modules import (
    _not_valid_config,
    config_readers,
    flat_complex_cfg,
    modules_category,
    read_from_yaml_config,
    )


@pytest.fixture(params=modules_category.items())
def module(request):
    """Give imported HADDOCK3 modules."""
    module_name, category = request.param
    mod = ".".join(['haddock', 'modules', category, module_name])
    module = importlib.import_module(mod)
    return module


def test_config_reader_can_read_defaults(module):
    """Test gear.config_reader can read modules' default file."""
    if module.DEFAULT_CONFIG.parent.name == 'topocg':
        read_from_yaml_config(module.DEFAULT_CONFIG)
    else:
        assert read_from_yaml_config(module.DEFAULT_CONFIG)


def test_all_defaults_have_the_same_name(module):
    """Test all default configuration files have the same name."""
    assert module.DEFAULT_CONFIG.name == 'defaults.yml'


complex_cfg = {
    "param1": {
        "default": 1,
        "other": None,
        "others": [None, None],
        },
    "param2": {
        "default": 2,
        "other": None,
        "others": [None, None],
        },
    "param3": {
        "param4": {
            "default": 4,
            "other": None,
            "others": [None, None],
            },
        },
    "param5": {
        "param6": {
            "param7": {
                "default": 7,
                "other": None,
                "others": [None, None],
                },
            },
        },
    }


complex_cfg_simplified = {
    "param1": 1,
    "param2": 2,
    "param3": {"param4": 4},
    "param5": {"param6": {"param7": 7}},
    }


def test_flat_complex_config_1():
    """Test if complex config is flatten properly."""
    result = flat_complex_cfg(complex_cfg)
    assert result == complex_cfg_simplified


def test_config_readers_keys():
    """Test config readers have all options."""
    assert set(config_readers.keys()) == {".yml", ".cfg"}


def test_not_valid_config():
    """Test not valid config error."""
    with pytest.raises(ConfigurationError):
        with _not_valid_config():
            config_readers[".zzz"]
