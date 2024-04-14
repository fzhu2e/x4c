import xarray as xr
import xesmf as xe
from . import utils

@xr.register_dataset_accessor('ice')
class IceDS:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def regrid(self, dlon=1, dlat=1, method='bilinear', periodic=True, grid='T'):
        ds = xr.Dataset()
        if grid == 'T':
            ds['lat'] = self._obj.TLAT
            ds['lon'] = self._obj.TLONG
        elif grid == 'U':
            ds['lat'] = self._obj.ULAT
            ds['lon'] = self._obj.ULONG
        else:
            raise ValueError('`grid` options: {"T", "U"}.')

        regridder = xe.Regridder(
            ds, xe.util.grid_global(dlon, dlat, cf=True, lon1=360),
            method=method, periodic=periodic,
        )

        ds_rgd = regridder(self._obj)
        return ds_rgd