"""
Create restructured text pages for module's default parameters.

These pages are then appended to the module's documentation page. See the
`.. include::` statement in the module's `.rst` file in the `docs/` folder.
The created files are save to `docs/modules/*/params/*.rst` files.

The pages generated by this script are not stagged to github. They are used
temporary just to create the HTML files for the documentation.
"""
import os
from collections.abc import Mapping
from pathlib import Path

from haddock import haddock3_repository_path, haddock3_source_path
#from haddock.modules import modules_category
#from haddock.modules.util import read_all_default_configs_yaml
from haddock.libs.libio import read_from_yaml


# prepare YAML markdown files
def main():
    """
    Prepare restructured text files from YAML default configs in modules.

    These files are written to the docs/ folder but not stagged to github.
    They are used only by Sphinx to generate the HTML documentation pages.
    """
    pattern = Path('modules', '*', '*', '*.yaml')
    configs = list(haddock3_source_path.glob(str(pattern)))

    for config in configs:

        module_name = config.parents[0].name
        category = config.parents[1].name
        params = read_from_yaml(config)

        text = build_rst(params)

        params_folder = Path(
            haddock3_repository_path,
            'docs',
            'modules',
            category,
            'params',
            )
        params_folder.mkdir(exist_ok=True)

        with open(Path(params_folder, f'{module_name}.rst'), 'w') as fout:
            fout.write(text)

    general_defaults = Path(haddock3_source_path, 'modules', 'defaults.yaml')
    general_params = read_from_yaml(general_defaults)
    text = build_rst(general_params)
    params_file = Path(
        haddock3_repository_path,
        'docs',
        'modules',
        'general_module_params.rst',
        )

    with open(params_file, 'w') as fout:
        fout.write(text)


def do_text(name, param):
    """Create text from parameter dictionary."""
    text = [
        f'{name}',
        f'{"`" * len(name)}',
        '',
        f'| *default*: {param["default"]!r}',
        f'| *type*: {param["type"]}',
        f'| *title*: {param["title"]}',
        f'| *short description*: {param["short"]}',
        f'| *long description*: {param["long"]}',
        f'| *group*: {param["group"]}',
        f'| *explevel*: {param["explevel"]}',
        '',
        ]

    return os.linesep.join(text)


def loop_params(config, easy, expert, guru):
    """Treat parameters for module."""
    # sort parameters by name
    sorted_ = sorted(
        ((k, v) for k, v in config.items()),
        key=lambda x: x[0],
        )

    for name, data in sorted_:
        if isinstance(data, Mapping) and "default" not in data:
            easy, exertp, guru = loop_params(data, easy, expert, guru)
            continue

        elif isinstance(data, Mapping):
            explevel = data["explevel"]

            text = do_text(name, data)

            if explevel == 'easy':
                easy.append(text)
            elif explevel == 'expert':
                expert.append(text)
            elif explevel == 'guru':
                guru.append(text)
            elif explevel == 'hidden':
                continue
            else:
                emsg = f'explevel {explevel!r} is not expected'
                raise AssertionError(emsg)

    easy.append('')
    expert.append('')
    guru.append('')

    return easy, expert, guru


def build_rst(module_params):
    """Build .rst text."""
    easy = ['Easy', '----', '']
    expert = ["Expert", '------', '']
    guru = ['Guru', '----', '']

    easy, expert, guru = loop_params(module_params, easy, expert, guru)

    doc = []
    for list_ in (easy, expert, guru):
        if len(list_) > 4:
            doc.extend(list_)

    text = os.linesep + os.linesep + os.linesep.join(doc)
    return text


if __name__ == "__main__":
    main()
