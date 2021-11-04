"""HADDOCK3 library."""
import logging
from os import get_terminal_size
from pathlib import Path

from haddock.libs.liblog import add_sysout_handler


log = logging.getLogger(__name__)
log.handlers.clear()
log.setLevel(logging.DEBUG)

try:
    get_terminal_size()
except OSError:
    has_terminal = False
else:
    has_terminal = True
    add_sysout_handler(log)

haddock3_source_path = Path(__file__).resolve().parent
toppar_path = Path(haddock3_source_path, "cns", "toppar")

FCC_path = Path(
    Path(__file__).resolve().parents[1],
    'fcc',
    )

v_major = "3"
v_minor = "0"
v_patch = "beta"
v_release = "unreleased"

current_version = f"{v_major}.{v_minor}.{v_patch}-{v_release}"
contact_us = 'https://github.com/haddocking/haddock3/issues'
