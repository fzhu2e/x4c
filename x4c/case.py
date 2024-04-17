import xarray as xr
import os
import glob
from . import core, utils, diags

class Timeseries:
    ''' Designed for postprocessed CESM timeseries
    '''
    def __init__(self, root_dir, grid_dict={'atm': 'ne16', 'lnd': 'ne16', 'rof': 'ne16', 'ocn': 'g16', 'ice': 'g16'},
                 path_pattern='comp/proc/tseries/month_1/casename.mdl.h_str.vn.timespan.nc'):
        self.root_dir = root_dir
        self.path_pattern = path_pattern
        self.grid_dict = grid_dict

        utils.p_header(f'>>> case.root_dir: {self.root_dir}')
        utils.p_header(f'>>> case.path_pattern: {self.path_pattern}')
        utils.p_header(f'>>> case.grid_dict: {self.grid_dict}')

        self.paths = glob.glob(
            os.path.join(
                self.root_dir,
                self.path_pattern \
                    .replace('comp', '**') \
                    .replace('casename', '*') \
                    .replace('mdl', '*') \
                    .replace('h_str', '*') \
                    .replace('vn', '*') \
                    .replace('timespan', '*'),
            )
        )

        self.ds = xr.Dataset()
        self.diags = {}
        self.vars_info = {}
        for path in self.paths:
            comp = path.split('/')[-5]
            mdl = path.split('.')[-5]
            h_str = path.split('.')[-4]
            vn = path.split('.')[-3]
            if vn not in self.vars_info:
                self.vars_info[vn] = (comp, mdl, h_str)

        utils.p_success(f'>>> case.vars_info created')

    def clear_ds(self, vn=None):
        if vn is not None:
            del(self.ds[vn])
        else:
            self.ds = xr.Dataset()

    def load(self, vn, adjust_month=True, load_idx=-1, regrid=False):
        if not isinstance(vn, (list, tuple)):
            vn = [vn]

        for v in vn:
            if v in self.vars_info and v not in self.ds:
                comp, mdl, h_str = self.vars_info[v]
                paths = sorted(glob.glob(
                    os.path.join(
                        self.root_dir,
                        self.path_pattern \
                            .replace('comp', comp) \
                            .replace('casename', '*') \
                            .replace('mdl', mdl) \
                            .replace('h_str', h_str) \
                            .replace('vn', v) \
                            .replace('timespan', '*'),
                    )
                ))
                if load_idx is not None:
                    ds =  core.load_dataset(paths[load_idx], adjust_month=adjust_month, comp=comp, grid=self.grid_dict[comp])
                else:
                    ds =  core.open_mfdataset(paths, adjust_month=adjust_month, comp=comp, grid=self.grid_dict[comp])

                self.ds[v] = ds.x[v] if not regrid else ds.x.regrid().x[v]
                self.ds.attrs.update(ds.x[v].attrs)
                utils.p_success(f'>>> case.ds["{v}"] created')

            elif v in ['KMT', 'z_t', 'z_w']:
                comp, mdl, h_str = self.vars_info['TEMP']
                paths = sorted(glob.glob(
                    os.path.join(
                        self.root_dir,
                        self.path_pattern \
                            .replace('comp', comp) \
                            .replace('casename', '*') \
                            .replace('mdl', mdl) \
                            .replace('h_str', h_str) \
                            .replace('vn', 'TEMP') \
                            .replace('timespan', '*'),
                    )
                ))
                with xr.open_dataset(paths[-1], decode_cf=False) as ds:
                    self.ds[v] = ds.x[v]
                
            elif v not in self.vars_info:
                utils.p_warning(f'>>> Variable {v} not existed')

            elif v in self.ds:
                utils.p_warning(f'>>> case.ds["{v}"] already loaded; to reload, run case.clear_ds("{v}") before case.load("{v}")')
        
    def calc(self, vn, load_idx=-1, adjust_month=True, **kws):
        plot_type, diag_name, mean_method = vn.split(':')
        func_name = f'calc_{plot_type}_{diag_name}'
        self.diags[vn] = diags.DiagCalc.__dict__[func_name](self, load_idx=load_idx, adjust_month=adjust_month, mean_method=mean_method, **kws)
        utils.p_success(f'>>> case.diags["{vn}"] created')

    def plot(self, vn, **kws):
        plot_type, diag_name, mean_method = vn.split(':')
        return diags.DiagPlot.__dict__[f'plot_{plot_type}'](self, diag_name, mean_method=mean_method, **kws)