import xarray as xr
import datetime

# from .atm import AtmDS
# from .ocn import OcnDS
# from .lnd import LndDS
# from .ice import IceDS
# from .rof import RofDS
from .core import XDataset, XDataArray
from . import utils

# get the version
from importlib.metadata import version
__version__ = version('xpp')

def load_dataset(*ps, adjust_month=False, comp=None, grid=None, **kws):
    ds = xr.load_dataset(*ps, **kws)
    if adjust_month:
        ds['time'] = ds['time'].get_index('time') - datetime.timedelta(days=1)

    ds.attrs['comp'] = comp
    ds.attrs['grid'] = grid
    return ds