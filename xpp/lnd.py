import xarray as xr
from . import utils


@xr.register_dataset_accessor('lnd')
class LndDS:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def regrid(self, weight_file=None):
        if weight_file is None:
            raise ValueError(f'`weight_file` must be specified')
        else:
            ds_rgd = utils.regrid_cam_se(self._obj, weight_file=weight_file)

        return ds_rgd
