"""
Test general implementation in haddock3 modules.

Ensures all modules follow the same compatible architecture.
"""
import importlib

import pytest

from haddock import modules_defaults_path
from haddock.core.exceptions import ConfigurationError
from haddock.gear.yaml2cfg import read_from_yaml_config
from haddock.libs.libio import read_from_yaml
from haddock.modules import (
    _not_valid_config,
    category_hierarchy,
    config_readers,
    modules_category,
    )


# defines which modules are already working
working_modules = [t for t in modules_category.items() if t[0] != 'topocg']

# errors message for missing params keys
ekmsg = '{!r} not in {!r} for {!r}'

# errors message for wrong param types
etmsg = 'Wrong type for {!r} in {!r}: got {!r}, expected {!r}'


# helper functions for tests below
def inspect_default_type(expected_type, value, param, module):
    """Inspect the default value is of expected type."""
    _vtype = type(value)
    assert _vtype in expected_type, \
        etmsg.format(param, module, _vtype, expected_type)


def ignore(*args, **kwargs):
    """Ignore this."""
    return


def inspect_commons(*args):
    """Inspect keys that are common to all parameters."""
    keys = (
        'title',
        'short',
        'long',
        'group',
        'explevel',
        )
    keys_inspect(keys, *args)


def inspect_int(*args):
    """Inspect keys that are needed for integer type parameters."""
    keys = ('min', 'max')
    keys_inspect(keys, *args)


def inspect_float(*args):
    """Inspect keys that are needed for float type parameters."""
    keys = ('min', 'max', 'precision')
    keys_inspect(keys, *args)


def inspect_list(*args):
    """Inspect keys that are needed for list type parameters."""
    keys = ('minitems', 'maxitems')
    keys_inspect(keys, *args)


def inspect_str(*args):
    """Inspect keys that are needed for string type parameters."""
    keys = ('minchars', 'maxchars')
    keys_inspect(keys, *args)


def keys_inspect(keys, d, param, module):
    """Loop over keys to inspect."""
    for key in keys:
        assert key in d, ekmsg.format(key, param, module)


# global dictionaries helping functions below
yaml_params_types = {
    "integer": (int, float),
    "float": (float,),
    "boolean": (bool,),
    "string": (str,),
    "list": (list,),
    "file": (str,),
    }


yaml_types_to_keys = {
    "integer": inspect_int,
    "float": inspect_float,
    "string": inspect_str,
    "list": inspect_list,
    "file": ignore,
    "boolean": ignore,
    }


@pytest.fixture(params=working_modules)
def modules(request):
    """Give imported HADDOCK3 modules."""
    module_name, category = request.param
    mod = ".".join(['haddock', 'modules', category, module_name])
    module = importlib.import_module(mod)
    return module


@pytest.fixture()
def module_yaml_dict(modules):
    """Give flatted yaml config dictionaries."""
    cfg = read_from_yaml_config(modules.DEFAULT_CONFIG)
    return cfg


@pytest.fixture()
def module_yaml_pure(modules):
    """Give the pure yaml dictionaries."""
    # this is the pure yaml config
    cfg = read_from_yaml(modules.DEFAULT_CONFIG)
    return cfg, modules


def test_config_reader_can_read_defaults(module_yaml_dict):
    """Test all yaml dictionaries are not empty."""
    assert module_yaml_dict


def test_general_config():
    """Test general config is readable and not empty."""
    assert read_from_yaml_config(modules_defaults_path)


def test_yaml_keys(module_yaml_pure):
    """
    Test keys in each parameter.

    Each parameter should have all the keys it is expected to have.
    """
    inspect_types(module_yaml_pure[0], module_yaml_pure[1].__name__)


def inspect_types(d, module):
    """Recursively inspect parameter keys according to their types."""
    # this dictionary also asserts that all types are controlled
    # it there's a type in the yaml that is not in this dict, a KeyError
    # raises

    for param, value in d.items():
        if isinstance(value, dict) and "default" in value:
            inspect_commons(value, param, module)
            _type = value["type"]
            inspect_default_type(
                yaml_params_types[_type],
                value["default"],
                param,
                module,
                )
            yaml_types_to_keys[_type](value, param, module)

        elif isinstance(value, dict):
            inspect_commons(value, param, module)

        else:
            assert False, (  # noqa: B011
                "If you reach here something changed in the defaults.yaml "
                "structure. Adapt the tests accordingly."
                )

    return


def test_all_defaults_have_the_same_name(modules):
    """Test all default configuration files have the same name."""
    assert modules.DEFAULT_CONFIG.name == 'defaults.yaml'


def test_config_readers_keys():
    """Test config readers have all options."""
    assert set(config_readers.keys()) == {".yaml", ".cfg"}


def test_not_valid_config():
    """Test not valid config error."""
    with pytest.raises(ConfigurationError):
        with _not_valid_config():
            config_readers[".zzz"]


def test_category_hierarchy():
    """Test all categories are listed in hierarchies."""
    categories_1 = set(category_hierarchy)
    assert len(categories_1) == len(category_hierarchy)

    categories_2 = set(modules_category.values())
    assert categories_1 == categories_2
