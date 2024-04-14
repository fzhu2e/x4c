import xarray as xr

from .atm import AtmDS
from .ocn import OcnDS
from .lnd import LndDS
from .ice import IceDS
from .rof import RofDS
from . import utils

# get the version
from importlib.metadata import version
__version__ = version('xpp')





def load_dataset(*ps, **kws):
    return xr.load_dataset(*ps, **kws)