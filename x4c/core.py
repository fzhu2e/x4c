import xarray as xr
import xesmf as xe
import numpy as np
from . import utils
import os
dirpath = os.path.dirname(__file__)

@xr.register_dataset_accessor('x')
class XDataset:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def regrid(self, dlon=1, dlat=1, weight_file=None, gs='T', method='bilinear', periodic=True):
        comp = self._obj.attrs['comp']
        grid = self._obj.attrs['grid']

        if grid in ['ne16']:
            # SE grid
            ds = self._obj.copy()
            if comp == 'lnd':
                ds = ds.rename_dims({'lndgrid': 'ncol'})
                
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

    def __getitem__(self, key):
        da = self._obj[key]
        da.attrs['source'] = self._obj.attrs['source']
        da.attrs['gw'] = self._obj['gw']
        da.attrs['lat'] = self._obj['lat']
        da.attrs['lon'] = self._obj['lon']
        return da


@xr.register_dataarray_accessor('x')
class XDataArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def annualize(self, months=None):
        da_ann = utils.annualize(self._obj, months=months)
        return da_ann

    def gm(self):
        da = self._obj
        gw = da.attrs['gw']
        da_gm = da.weighted(gw).mean(list(gw.dims))
        return da_gm

    def nhm(self):
        da = self._obj
        gw = da.attrs['gw']
        lat = da.attrs['lat']
        da_nhm = da.where(lat>0).weighted(gw).mean(list(gw.dims))
        return da_nhm

    def shm(self):
        da = self._obj
        gw = da.attrs['gw']
        lat = da.attrs['lat']
        da_shm = da.where(lat<0).weighted(gw).mean(list(gw.dims))
        return da_shm

    def geo_mean(self, ind=None, lat_min=-90, lat_max=90, lon_min=0, lon_max=360):
        da = self._obj
        if ind is None:
            dam = utils.geo_mean(da, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        elif ind == 'nino3.4':
            dam = utils.geo_mean(da, lat_min=-5, lat_max=5, lon_min=np.mod(-170, 360), lon_max=np.mod(-120, 360))
        elif ind == 'nino1+2':
            dam = utils.geo_mean(da, lat_min=-10, lat_max=10, lon_min=np.mod(-90, 360), lon_max=np.mod(-80, 360))
        elif ind == 'nino3':
            dam = utils.geo_mean(da, lat_min=-5, lat_max=5, lon_min=np.mod(-150, 360), lon_max=np.mod(-90, 360))
        elif ind == 'nino4':
            dam = utils.geo_mean(da, lat_min=-5, lat_max=5, lon_min=np.mod(160, 360), lon_max=np.mod(-150, 360))
        elif ind == 'wpi':
            # Western Pacific Index
            dam = utils.geo_mean(da, lat_min=-10, lat_max=10, lon_min=np.mod(120, 360), lon_max=np.mod(150, 360))
        elif ind == 'tpi':
            # Tri-Pole Index
            v1 = utils.geo_mean(da, lat_min=25, lat_max=45, lon_min=np.mod(140, 360), lon_max=np.mod(-145, 360))
            v2 = utils.geo_mean(da, lat_min=-10, lat_max=10, lon_min=np.mod(170, 360), lon_max=np.mod(-90, 360))
            v3 = utils.geo_mean(da, lat_min=-50, lat_max=-15, lon_min=np.mod(150, 360), lon_max=np.mod(-160, 360))
            dam = v2 - (v1 + v3)/2
        elif ind == 'dmi':
            # Indian Ocean Dipole Mode
            dmiw = utils.geo_mean(da, lat_min=-10, lat_max=10, lon_min=50 ,lon_max=70)
            dmie = utils.geo_mean(da,lat_min=-10,lat_max=0,lon_min=90,lon_max=110)
            dam = dmiw - dmie
        elif ind == 'iobw':
            # Indian Ocean Basin Wide
            dam =  utils.geo_mean(da, lat_min=-20, lat_max=20, lon_min=40 ,lon_max=100)
        else:
            raise ValueError('`ind` options: {"nino3.4", "nino1+2", "nino3", "nino4", "wpi", "tpi", "dmi", "iobw"}')

        return dam
    
    def zm(self):
        da = self._obj
        da_zm = da.mean(('time', 'lon'))
        return da_zm