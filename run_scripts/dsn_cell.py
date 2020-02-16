from typing import Mapping, Any

import argparse

from bag.io import read_yaml
from bag.core import BagProject
from bag.util.misc import register_pdb_hook

from bag.simulation.design import DesignerBase

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Design cell from spec file.')
    parser.add_argument('specs', help='Design specs file name.')
    # parser.add_argument('-x', '--extract', action='store_true', default=False,
    #                     help='Run extracted simulation')
    # parser.add_argument('-f', '--force_extract', action='store_true', default=False,
    #                     help='Force RC extraction even if layout/schematic are unchanged')
    parser.add_argument('-ic', '--ignore_cache', action='store_true', default=False,
                        help='Force Design even if design specs have not changed')
    # parser.add_argument('-c', '--gen_sch', action='store_true', default=False,
    #                     help='Generate testbench schematics for debugging.')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    specs: Mapping[str, Any] = read_yaml(args.specs)

    DesignerBase.design_cell(prj, specs, use_cache=not args.ignore_cache)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['bprj']

    run_main(_prj, _args)
