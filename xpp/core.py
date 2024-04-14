import xarray as xr
import xesmf as xe
from . import utils
import os
dirpath = os.path.dirname(__file__)

@xr.register_dataset_accessor('xpp')
class XDataset:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def regrid(self, dlon=1, dlat=1, weight_file=None, gs='T', method='bilinear', periodic=True):
        comp = self._obj.attrs['comp']
        grid = self._obj.attrs['grid']

        if grid in ['ne16']:
            # SE grid
            if comp == 'lnd':
                ds = self._obj.copy()
                ds = ds.rename_dims({'lndgrid': 'ncol'})
                # ds['lat'] = self._obj['lat_lnd']
                # ds['lon'] = self._obj['lon_lnd']
                
            if weight_file is not None:
                ds_rgd = utils.regrid_cam_se(ds, weight_file=weight_file)
            else:
                ds_rgd = utils.regrid_cam_se(ds, weight_file=os.path.join(dirpath, f'./regrid_wgts/map_{grid}np4_TO_{dlon}x{dlat}d_aave.nc'))

        elif grid[0] == 'g':
            # ocn grid
            ds = xr.Dataset()
            if gs == 'T':
                ds['lat'] = self._obj.TLAT
                if comp == 'ice':
                    ds['lon'] = self._obj.TLON
                else:
                    ds['lon'] = self._obj.TLONG
            elif gs == 'U':
                ds['lat'] = self._obj.ULAT
                if comp == 'ice':
                    ds['lon'] = self._obj.ULON
                else:
                    ds['lon'] = self._obj.ULONG
            else:
                raise ValueError('`gs` options: {"T", "U"}.')

            regridder = xe.Regridder(
                ds, xe.util.grid_global(dlon, dlat, cf=True, lon1=360),
                method=method, periodic=periodic,
            )

            ds_rgd = regridder(self._obj)

        else:
            raise ValueError(f'grid [{grid}] is not supported; please submit an issue on Github to make a request.')


        return ds_rgd

    def annualize(self, months=None):
        da_ann = utils.annualize(self._obj, months=months)
        return da_ann

@xr.register_dataarray_accessor('xpp')
class XDataArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def annualize(self, months=None):
        da_ann = utils.annualize(self._obj, months=months)
        return da_ann
