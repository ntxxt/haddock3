"""CNS scripts util functions."""
import itertools
import math
from functools import partial
from os import linesep
from pathlib import Path

from haddock import EmptyPath, log
from haddock.core import cns_paths
from haddock.libs import libpdb
from haddock.libs.libfunc import false, true
from haddock.libs.libmath import RandomNumberGenerator
from haddock.libs.libontology import PDBFile
from haddock.libs.libutil import transform_to_list


RND = RandomNumberGenerator()


def generate_default_header(path=None):
    """Generate CNS default header."""
    if path is not None:
        axis = load_axis(**cns_paths.get_axis(path))
        link = load_link(Path(path, cns_paths.LINK_FILE))
        scatter = load_scatter(Path(path, cns_paths.SCATTER_LIB))
        tensor = load_tensor(**cns_paths.get_tensors(path))
        trans_vec = load_trans_vectors(**cns_paths.get_translation_vectors(path))  # noqa: E501
        water_box = load_boxtyp20(cns_paths.get_water_box(path)["boxtyp20"])

    else:
        axis = load_axis(**cns_paths.axis)
        link = load_link(cns_paths.link_file)
        scatter = load_scatter(cns_paths.scatter_lib)
        tensor = load_tensor(**cns_paths.tensors)
        trans_vec = load_trans_vectors(**cns_paths.translation_vectors)
        water_box = load_boxtyp20(cns_paths.water_box["boxtyp20"])

    return (
        link,
        trans_vec,
        tensor,
        scatter,
        axis,
        water_box,
        )


def _is_nan(x):
    """Inspect if is nan."""
    try:
        return math.isnan(x)
    except (ValueError, TypeError):
        return False


def filter_empty_vars(v):
    """
    Filter empty variables.

    See: https://github.com/haddocking/haddock3/issues/162

    Returns
    -------
    bool
        Returns `True` if the variable is not empty, and `False` if
        the variable is empty. That is, `False` reflects those variables
        that should not be written in CNS.

    Raises
    ------
    TypeError
        If the type of `value` is not supported by CNS.
    """
    cases = (
        (lambda x: _is_nan(x), false),
        (lambda x: isinstance(x, str) and bool(x), true),
        (lambda x: isinstance(x, str) and not bool(x), false),
        (lambda x: isinstance(x, bool), true),  # it should return True
        (lambda x: isinstance(x, (EmptyPath, Path)), true),
        (lambda x: type(x) in (int, float), true),
        (lambda x: x is None, false),
        )

    for detect, give in cases:
        if detect(v):
            return give(v)
    else:
        emsg = f"Value {v!r} has a unknown type for CNS: {type(v)}."
        log.error(emsg)
        raise TypeError(emsg)


def load_workflow_params(
        param_header=f"{linesep}! Parameters{linesep}",
        **params,
        ):
    """
    Write the values at the header section.

    "Empty variables" are ignored. These are defined accoring to
    :func:`filter_empty_vars`.

    Parameters
    ----------
    params : dict
        Dictionary containing the key:value pars for the parameters to
        be written to CNS. Values cannot be of dictionary type.

    Returns
    -------
    str
        The string with the CNS parameters defined.
    """
    non_empty_parameters = (
        (k, v) for k, v in params.items() if filter_empty_vars(v)
        )

    # types besides the ones in the if-statements should not enter this loop
    for param, v in non_empty_parameters:
        param_header += write_eval_line(param, v)

    assert isinstance(param_header, str)
    return param_header


def write_eval_line(param, value, eval_line="eval (${}={})"):
    """Write the CNS eval line depending on the type of `value`."""
    eval_line += linesep

    if isinstance(value, bool):
        value = str(value).lower()
        return eval_line.format(param, value)

    elif isinstance(value, str):
        value = '"' + value + '"'
        return eval_line.format(param, value)

    elif isinstance(value, Path):
        value = '"' + str(value) + '"'
        return eval_line.format(param, value)

    elif isinstance(value, EmptyPath):
        return eval_line.format(param, '""')

    elif isinstance(value, (int, float)):
        return eval_line.format(param, value)

    else:
        emsg = f"Unexpected type when writing CNS header: {type(value)}"
        log.error(emsg)
        raise TypeError(emsg)


def load_link(mol_link):
    """Add the link header."""
    return load_workflow_params(
        param_header=f"{linesep}! Link file{linesep}",
        link_file=mol_link)


load_axis = partial(load_workflow_params, param_header=f"{linesep}! Axis{linesep}")  # noqa: E501
load_tensor = partial(load_workflow_params, param_header=f"{linesep}! Tensors{linesep}")  # noqa: E501
prepare_output = partial(load_workflow_params, param_header=f"{linesep}! Output structure{linesep}")  # noqa: E501
load_trans_vectors = partial(load_workflow_params, param_header=f"{linesep}! Translation vectors{linesep}")  # noqa: E501

load_ambig = partial(write_eval_line, "ambig_fname")
load_unambig = partial(write_eval_line, "unambig_fname")
load_hbond = partial(write_eval_line, "hbond_fname")
load_dihe = partial(write_eval_line, "dihe_f")
load_tensor_tbl = partial(write_eval_line, "tensor_tbl")


def load_scatter(scatter_lib):
    """Add scatter library."""
    return load_workflow_params(
        param_header=f"{linesep}! Scatter lib{linesep}",
        scatter_lib=scatter_lib)


def load_boxtyp20(waterbox_param):
    """Add boxtyp20 eval line."""
    return load_workflow_params(
        param_header=f"{linesep}! Water box{linesep}",
        boxtyp20=waterbox_param)


# This is used by docking
def prepare_multiple_input(pdb_input_list, psf_input_list):
    """Prepare multiple input files."""
    input_str = f"{linesep}! Input structure{linesep}"
    for psf in psf_input_list:
        input_str += f"structure{linesep}"
        input_str += f"  @@{psf}{linesep}"
        input_str += f"end{linesep}"

    ncount = 1
    for pdb in pdb_input_list:
        input_str += f"coor @@{pdb}{linesep}"
        input_str += write_eval_line(f'input_pdb_filename_{ncount}', pdb)
        ncount += 1

    # check how many chains there are across all the PDBs
    chain_l = []
    for pdb in pdb_input_list:
        for element in libpdb.identify_chainseg(pdb):
            chain_l.append(element)
    ncomponents = len(set(itertools.chain(*chain_l)))
    input_str += write_eval_line('ncomponents', ncomponents)

    seed = RND.randint(100, 999)
    input_str += write_eval_line('seed', seed)

    return input_str


# This is used by Topology and Scoring
def prepare_single_input(pdb_input, psf_input=None):
    """Input of the CNS file.

    This section will be written for any recipe even if some CNS variables
    are not used, it should not be an issue.
    """
    input_str = f"{linesep}! Input structure{linesep}"

    if psf_input:
        # if isinstance(psf_input, str):
        input_str += f"structure{linesep}"
        input_str += f"  @@{psf_input}{linesep}"
        input_str += f"end{linesep}"
        input_str += f"coor @@{pdb_input}{linesep}"
        if isinstance(psf_input, list):
            input_str += f"structure{linesep}"
            for psf in psf_input:
                input_str += f"  @@{psf}{linesep}"
            input_str += f"end{linesep}"

    # $file variable is still used by some CNS recipes, need refactoring!
    input_str += write_eval_line('file', pdb_input)
    segids, chains = libpdb.identify_chainseg(pdb_input)
    chainsegs = sorted(list(set(segids) | set(chains)))

    ncomponents = len(chainsegs)
    input_str += write_eval_line("ncomponents", ncomponents)

    for i, segid in enumerate(chainsegs, start=1):
        input_str += write_eval_line(f"prot_segid_{i}", segid)

    seed = RND.randint(100, 99999)
    input_str += write_eval_line('seed', seed)

    return input_str


def prepare_cns_input(
        model_number,
        input_element,
        step_path,
        recipe_str,
        defaults,
        identifier,
        ambig_fname="",
        native_segid=False,
        default_params_path=None,
        ):
    """
    Generate the .inp file needed by the CNS engine.

    Parameters
    ----------
    model_number : int
        The number of the model. Will be used as file name suffix.

    input_element : `libs.libontology.Persisten`, list of those
    """
    # read the default parameters
    default_params = load_workflow_params(**defaults)
    default_params += write_eval_line('ambig_fname', ambig_fname)

    # write the PDBs
    pdb_list = [
        pdb.rel_path
        for pdb in transform_to_list(input_element)
        ]

    # write the PSFs
    psf_list = []
    if isinstance(input_element, (list, tuple)):
        for pdb in input_element:
            if isinstance(pdb.topology, (list, tuple)):
                for psf in pdb.topology:
                    psf_fname = psf.rel_path
                    psf_list.append(psf_fname)
            else:
                psf_fname = pdb.topology.rel_path
                psf_list.append(psf_fname)

    elif isinstance(input_element.topology, (list, tuple)):
        pdb = input_element  # for clarity
        for psf in pdb.topology:
            psf_fname = psf.rel_path
            psf_list.append(psf_fname)
    else:
        pdb = input_element  # for clarity
        psf_fname = pdb.topology.rel_path
        psf_list.append(psf_fname)

    input_str = prepare_multiple_input(pdb_list, psf_list)

    output_pdb_filename = f"{identifier}_{model_number}.pdb"

    output = f"{linesep}! Output structure{linesep}"
    output += write_eval_line('output_pdb_filename', output_pdb_filename)

    # prepare chain/seg IDs
    segid_str = ""
    if native_segid:
        chainid_list = []
        if isinstance(input_element, (list, tuple)):
            for pdb in input_element:

                segids, chains = \
                    libpdb.identify_chainseg(pdb.rel_path, sort=False)

                chainsegs = sorted(list(set(segids) | set(chains)))
                chainid_list.extend(chainsegs)

            for i, _chainseg in enumerate(chainid_list, start=1):
                segid_str += write_eval_line(f'prot_segid_{i}', _chainseg)

        else:
            segids, chains = \
                libpdb.identify_chainseg(input_element.rel_path, sort=False)

            chainsegs = sorted(list(set(segids) | set(chains)))

            for i, _chainseg in enumerate(chainsegs, start=1):
                segid_str += write_eval_line(f'prot_segid_{i}', _chainseg)

    output += write_eval_line('count', model_number)

    inp = (
        default_params
        + input_str
        + output
        + segid_str
        + recipe_str
        )

    inp_file = Path(f"{identifier}_{model_number}.inp")
    inp_file.write_text(inp)
    return inp_file


def prepare_expected_pdb(model_obj, model_nb, path, identifier):
    """Prepare a PDBobject."""
    expected_pdb_fname = Path(path, f"{identifier}_{model_nb}.pdb")
    pdb = PDBFile(expected_pdb_fname, path=path)
    if type(model_obj) == tuple:
        pdb.topology = [p.topology for p in model_obj]
    else:
        pdb.topology = model_obj.topology
    return pdb
