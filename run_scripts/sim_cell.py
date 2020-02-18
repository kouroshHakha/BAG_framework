import argparse
from argparse import Namespace
from pathlib import Path
import pdb

from bag.io.file import Pickle, Yaml
from bag.core import BagProject
from bag.util.misc import register_pdb_hook

register_pdb_hook()

io_cls_dict = {
    'pickle': Pickle,
    'yaml': Yaml,
}


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('specs_fname', help='specs yaml file')
    parser.add_argument('--no-cell', dest='gen_cell', action='store_false',
                        default=True, help='skip cell generation')
    parser.add_argument('--no-wrapper', dest='gen_wrapper', action='store_false',
                        default=True,  help='skip wrapper generation')
    parser.add_argument('--no-tb', dest='gen_tb', action='store_false',
                        default=True,  help='skip tb generation')
    parser.add_argument('--load', dest='load_results', action='store_true',
                        default=False,  help='skip simulation, just load the results')
    parser.add_argument('-x', '--extract', dest='extract', action='store_true',
                        default=False, help='do extracted simulation')
    parser.add_argument('--no-sim', dest='run_sim', action='store_false',
                        default=True, help='run simulation, --load has a priority over this')
    parser.add_argument('--format', default='yaml',
                        help='format of spec file (yaml, json, pickle)')
    parser.add_argument('-dump', '--dump', default='', help='output will be dumped to this path, '
                                                            'according to the format specified')
    parser.add_argument('--pause', default=False, action='store_true',
                        help='True to pause using pdb.set_trace() after simulation is done')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: Namespace):
    specs_fname = Path(args.specs_fname)
    io_cls = io_cls_dict[args.format]
    specs = io_cls.load(str(specs_fname))

    results = prj.simulate_cell(specs=specs,
                                gen_cell=args.gen_cell,
                                gen_wrapper=args.gen_wrapper,
                                gen_tb=args.gen_tb,
                                load_results=args.load_results,
                                extract=args.extract,
                                run_sim=args.run_sim)

    if args.pause:
        pdb.set_trace()

    if results is not None and args.dump:
        out_tmp_file = Path(args.dump)
        io_cls.save(results, out_tmp_file)


if __name__ == '__main__':

    args = parse_args()
    local_dict = locals()
    bprj = local_dict.get('bprj', BagProject())
    run_main(bprj, args)
