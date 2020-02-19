from __future__ import annotations

from typing import Any, Union, Type, Mapping, Dict, Optional, cast

import abc
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
import pprint

from ..concurrent.core import batch_async_task
from ..util.immutable import to_immutable
from ..util.importlib import import_class
from ..io.file import Yaml

from ..core import BagProject


@dataclass
class DUTInfo:
    impl_lib: str
    impl_cell: str
    view: str


@dataclass
class SchGenInfo:
    sch_lib: str
    sch_cell: str


class DesignerBase(abc.ABC):
    """Base class of all design scripts.

    dsn_specs is a private attribute. It can be set either during initialization or through
    update() method. However it can be read as an immutable type.
    """

    def __init__(self,  bprj: BagProject, root_dir: Union[Path, str], dsn_specs: Mapping[str, Any],
                 dsn_db: Optional[Dict[int, Mapping[str, Any]]] = None, use_cache: bool = True
                 ) -> None:
        # Bag 2.0 has a bad architecture in terms of lay_db, sch_db. prj only owns sch_db and
        # not lay_db. lay_db should be passed around directly (and is synced with impl_lib ) but
        # sch_db can be passed around by passing the prj object around.

        self._prj = bprj
        self._root_dir = Path(root_dir)
        self._work_dir = None
        self._dsn_specs = {k: deepcopy(v) for k, v in dsn_specs.items()}
        self.specs = None

        if dsn_db is None:
            if use_cache:
                try:
                    self._dsn_db = Yaml.load(self._root_dir / 'dsn_cache.yaml')
                except FileNotFoundError:
                    self._dsn_db = {}
            else:
                self._dsn_db = {}
        else:
            self._dsn_db = dsn_db

        self.commit()

    @classmethod
    @abc.abstractmethod
    def get_params_info(cls) -> Dict[str, str]:
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        raise NotImplementedError

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @classmethod
    def get_sch_gen_info(cls) -> Optional[SchGenInfo]:
        return None

    @classmethod
    def get_lay_gen_cls(cls) -> Optional[str]:
        return None

    def _save_cache(self):
        print(f'Saving dsn_db cache to {str(self._root_dir)}...')
        Yaml.save(self._dsn_db, self._root_dir / 'dsn_cache.yaml')

    @property
    def key(self):
        return to_immutable(dict(cls_name=self.__class__.__name__, specs=self.specs))

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return {}

    def update(self, specs):
        # updates dsn_specs and commit changes
        self._dsn_specs = {k: deepcopy(v) for k, v in specs.items()}
        self.commit()

    def new_designer(self, dsn_cls: Union[str, Type[DesignerBase]],
                     dsn_specs: Mapping[str, Any]):
        # call this to create new designer masters
        if isinstance(dsn_cls, str):
            dsn_cls = cast(Type[DesignerBase], import_class(dsn_cls))
        return dsn_cls(self._prj, self._root_dir, dsn_specs, self._dsn_db)

    def design(self) -> Mapping[str, Any]:
        # This is the public method that should be called when doing hierarchy design
        coro = self.async_design()
        results = batch_async_task([coro])
        if results is None:
            raise ValueError('Design script cancelled.')

        ans = results[0]
        if isinstance(ans, Exception):
            raise ans
        return ans

    async def async_design(self):
        # This is the public method that should be called when doing hierarchy design, via asyncio
        try:
            ans = self._dsn_db[hash(self.key)]
            print('Design found, loading answer ...')
        except KeyError:
            ans = await self._async_design()
            self._dsn_db[hash(self.key)] = ans
        return ans

    @abc.abstractmethod
    async def _async_design(self) -> Mapping[str, Any]:
        # abstract method: design methodology goes here
        pass

    async def async_gen_dut(self, cell_name: str, dut_specs: Mapping[str, Any],
                            lay_cls: Optional[str] = None, sch_gen: SchGenInfo = None,
                            extract: bool = False):
        # creates a dut a single dut can be shared for multiple simulations / measurements
        impl_lib = self.get_default_impl_lib()
        gen_specs = dict(
            impl_lib=impl_lib,
            impl_cell=cell_name,
            params=dut_specs
        )
        if lay_cls:
            gen_specs['lay_cls'] = lay_cls
        if sch_gen:
            gen_specs['sch_lib'] = sch_gen.sch_lib
            gen_specs['sch_cell'] = sch_gen.sch_cell

        self._prj.generate_cell(gen_specs, gen_lay=extract, gen_sch=True, run_lvs=extract,
                                run_rcx=extract)
        view = 'netlist' if extract else 'schematic'
        return DUTInfo(impl_lib, cell_name, view)

    async def aync_sim_cell(self, dut: DUTInfo, sim_specs: Mapping[str, Any],
                            is_tbm: bool = False) -> Mapping[str, Any]:
        # runs tbm / simulation and returns the raw data
        sim_cell_specs = dict(
            impl_lib=dut.impl_lib,
            impl_cell=dut.impl_cell,
            root_dir=self._root_dir / self.__class__.__name__,
        )
        if is_tbm:
            sim_cell_specs['tbm_specs'] = sim_specs
        else:
            sim_cell_specs['sim_params'] = sim_specs

        extract = dut.view == 'netlist'
        res = cast(Mapping[str, Any], self._prj.simulate_cell(sim_cell_specs, gen_cell=False,
                                                              extract=extract))
        return res

    async def async_meas_cell(self, dut: DUTInfo,
                              meas_specs: Mapping[str, Any]) -> Mapping[str, Any]:
        # runs the measurement class and returns the result dictionary
        meas_cell_specs = dict(
            impl_lib=dut.impl_lib,
            impl_cell=dut.impl_cell,
            root_dir=self._root_dir / self.__class__.__name__,
            mm_specs=meas_specs
        )

        extract = dut.view == 'netlist'
        return self._prj.measure_cell(meas_cell_specs, gen_cell=False, extract=extract)

    def get_default_impl_lib(self):
        return self._root_dir.stem

    def get_design_dir(self, parent_dir: Path) -> Path:
        return parent_dir / 'schematic'

    def commit(self) -> None:
        """Commit changes to specs dictionary.  Perform necessary initialization."""
        for k, v in self.get_default_param_values().items():
            if k not in self._dsn_specs:
                self._dsn_specs[k] = v

        self._work_dir = self.get_design_dir(self._root_dir / self.__class__.__name__)
        self.specs = to_immutable(self._dsn_specs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save_cache()

    @classmethod
    def design_cell(cls, prj: BagProject, specs: Mapping[str, Any], use_cache: bool = True) -> None:
        # used for running top-level design script
        dsn_str: Union[str, Type[DesignerBase]] = specs['dsn_class']
        root_dir: Union[str, Path] = specs['root_dir']
        dsn_params: Mapping[str, Any] = specs['dsn_params']

        dsn_cls = cast(Type[DesignerBase], import_class(dsn_str))
        if isinstance(root_dir, str):
            root_path = Path(root_dir)
        else:
            root_path = root_dir

        with dsn_cls(prj, root_path, dsn_params, use_cache=use_cache) as designer:
            summary = designer.design()
        pprint.pprint(summary)
