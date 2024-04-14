import xarray as xr
import datetime

from .core import XDataset, XDataArray
from . import utils

# get the version
from importlib.metadata import version
__version__ = version('x4c')

def load_dataset(path, adjust_month=False, comp=None, grid=None, **kws):
    ds = xr.load_dataset(path, **kws)
    if adjust_month:
        ds['time'] = ds['time'].get_index('time') - datetime.timedelta(days=1)

    ds.attrs['comp'] = comp
    ds.attrs['grid'] = grid

    grid_weight_dict = {
        'atm': 'area',
        'ocn': 'TAREA',
        'ice': 'tarea',
        'lnd': 'area',
    }

    lon_dict = {
        'atm': 'lon',
        'ocn': 'TLONG',
        'ice': 'TLON',
        'lnd': 'lon',
    }

    lat_dict = {
        'atm': 'lat',
        'ocn': 'TLAT',
        'ice': 'TLAT',
        'lnd': 'lat',
    }

    ds.attrs['source'] = path
    ds['gw'] = ds[grid_weight_dict[comp]]
    ds['lat'] = ds[lat_dict[comp]]
    ds['lon'] = ds[lon_dict[comp]]
    return ds