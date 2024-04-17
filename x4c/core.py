import xarray as xr
xr.set_options(keep_attrs=True)

import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


from . import utils, visual
import os
dirpath = os.path.dirname(__file__)

def load_dataset(path, adjust_month=False, comp=None, grid=None, lazyload=False, **kws):
    if not lazyload:
        ds = xr.load_dataset(path, **kws)
    else:
        ds = xr.open_dataset(path, **kws)
    ds = utils.update_ds(ds, path=path, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

def open_mfdataset(paths, adjust_month=False, comp=None, grid=None, **kws):
    ds = xr.open_mfdataset(paths, **kws)
    ds = utils.update_ds(ds, path=paths, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

@xr.register_dataset_accessor('x')
class XDataset:
    def __init__(self, ds):
        self.ds = ds

    def regrid(self, dlon=1, dlat=1, weight_file=None, gs='T', method='bilinear', periodic=True):
        comp = self.ds.attrs['comp']
        grid = self.ds.attrs['grid']

        if grid in ['ne16']:
            # SE grid
            ds = self.ds.copy()
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
                ds['lat'] = self.ds.TLAT
                if comp == 'ice':
                    ds['lon'] = self.ds.TLON
                else:
                    ds['lon'] = self.ds.TLONG
            elif gs == 'U':
                ds['lat'] = self.ds.ULAT
                if comp == 'ice':
                    ds['lon'] = self.ds.ULON
                else:
                    ds['lon'] = self.ds.ULONG
            else:
                raise ValueError('`gs` options: {"T", "U"}.')

            regridder = xe.Regridder(
                ds, xe.util.grid_global(dlon, dlat, cf=True, lon1=360),
                method=method, periodic=periodic,
            )

            ds_rgd = regridder(self.ds, keep_attrs=True)

        else:
            raise ValueError(f'grid [{grid}] is not supported; please submit an issue on Github to make a request.')

        ds_rgd.attrs = dict(self.ds.attrs)
        return ds_rgd

    def annualize(self, months=None):
        ds_ann = utils.annualize(self.ds, months=months)
        ds_ann.attrs = dict(self.ds.attrs)
        return ds_ann

    def __getitem__(self, key):
        da = self.ds[key]

        if 'source' in self.ds.attrs:
            da.attrs['source'] = self.ds.attrs['source']

        if 'gw' in self.ds:
            da.attrs['gw'] = self.ds['gw']

        if 'lat' in self.ds:
            da.attrs['lat'] = self.ds['lat']

        if 'lon' in self.ds:
            da.attrs['lon'] = self.ds['lon']

        if 'comp' in self.ds.attrs:
            da.attrs['comp'] = self.ds.attrs['comp']
            if 'time' in da.coords:
                da.time.attrs['long_name'] = 'Model Year'
        if 'grid' in self.ds.attrs:
            da.attrs['grid'] = self.ds.attrs['grid']

        return da


@xr.register_dataarray_accessor('x')
class XDataArray:
    def __init__(self, da):
        self.da = da

    def annualize(self, months=None):
        da = utils.annualize(self.da, months=months)
        da = utils.update_attrs(da, self.da)
        return da

    @property
    def gm(self):
        gw = self.da.attrs['gw']
        da = self.da.weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        return da

    @property
    def nhm(self):
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat>0).weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        return da

    @property
    def shm(self):
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat<0).weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        return da

    @property
    def zm(self):
        if 'time' in self.da.coords:
            da = self.da.mean(('time', 'lon'))
        else:
            da = self.da.mean('lon')
        da = utils.update_attrs(da, self.da)
        return da

    def geo_mean(self, ind=None, lat_min=-90, lat_max=90, lon_min=0, lon_max=360):
        if ind is None:
            da = utils.geo_mean(self.da, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        elif ind == 'nino3.4':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(-170, 360), lon_max=np.mod(-120, 360))
        elif ind == 'nino1+2':
            da = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(-90, 360), lon_max=np.mod(-80, 360))
        elif ind == 'nino3':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(-150, 360), lon_max=np.mod(-90, 360))
        elif ind == 'nino4':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(160, 360), lon_max=np.mod(-150, 360))
        elif ind == 'wpi':
            # Western Pacific Index
            da = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(120, 360), lon_max=np.mod(150, 360))
        elif ind == 'tpi':
            # Tri-Pole Index
            v1 = utils.geo_mean(self.da, lat_min=25, lat_max=45, lon_min=np.mod(140, 360), lon_max=np.mod(-145, 360))
            v2 = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(170, 360), lon_max=np.mod(-90, 360))
            v3 = utils.geo_mean(self.da, lat_min=-50, lat_max=-15, lon_min=np.mod(150, 360), lon_max=np.mod(-160, 360))
            da = v2 - (v1 + v3)/2
        elif ind == 'dmi':
            # Indian Ocean Dipole Mode
            dmiw = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=50 ,lon_max=70)
            dmie = utils.geo_mean(self.da,lat_min=-10,lat_max=0,lon_min=90,lon_max=110)
            da = dmiw - dmie
        elif ind == 'iobw':
            # Indian Ocean Basin Wide
            da =  utils.geo_mean(self.da, lat_min=-20, lat_max=20, lon_min=40 ,lon_max=100)
        else:
            raise ValueError('`ind` options: {"nino3.4", "nino1+2", "nino3", "nino4", "wpi", "tpi", "dmi", "iobw"}')

        da.attrs = dict(self.da.attrs)
        if 'comp' in da.attrs and 'time' in da.coords:
            da.time.attrs['long_name'] = 'Model Year'
        return da

    def plot(self, title=None, figsize=None, ax=None, latlon_range=None,
             projection='Robinson', transform='PlateCarree', central_longitude=180, proj_args=None,
             add_gridlines=False, gridline_labels=True, gridline_style='--', ssv=None, coastline_zorder=99, coastline_width=1,
             **kws):
        ndim = len(self.da.dims)
        if ndim == 2 and 'lat' in self.da.coords and 'lon' in self.da.coords:
            # map
            if ax is None:
                if figsize is None: figsize = (10, 3)
                fig = plt.figure(figsize=figsize)
                proj_args = {} if proj_args is None else proj_args
                proj_args_default = {'central_longitude': central_longitude}
                proj_args_default.update(proj_args)
                _projection = ccrs.__dict__[projection](**proj_args_default)
                _transform = ccrs.__dict__[transform]()
                ax = plt.subplot(projection=_projection)

            _plt_kws = {
                'transform': _transform,
                'extend': 'both',
                'cmap': visual.infer_cmap(self.da),
                'cbar_kwargs': {
                    'label': f'{self.da.name} [{self.da.units}]',
                    'aspect': 10,
                },
            }
            _plt_kws = utils.update_dict(_plt_kws, kws)

            if latlon_range is not None:
                lat_min, lat_max, lon_min, lon_max = latlon_range
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=_transform)

            if add_gridlines:
                gl = ax.gridlines(linestyle=gridline_style, draw_labels=gridline_labels)
                gl.top_labels = False
                gl.right_labels = False

            if self.da.attrs['comp'] in ['ocn', 'ice']:
                ax.contour(self.da.lon, self.da.lat, np.isnan(self.da), levels=[0, 1], colors='k', transform=_transform, zorder=coastline_zorder, linewidths=coastline_width)
            else:
                if ssv is None:
                    ax.coastlines(zorder=coastline_zorder, linewidth=coastline_width)
                else:
                    ax.contour(ssv.lon, ssv.lat, np.isnan(ssv), levels=[0, 1], colors='k', transform=_transform, zorder=coastline_zorder, linewidths=coastline_width)

            self.da.plot.contourf(ax=ax, **_plt_kws)


        elif ndim == 2:
            # vertical
            if figsize is None: figsize = (6, 3)
            if ax is None: fig, ax = plt.subplots(figsize=figsize)
            _plt_kws = {
                'extend': 'both',
                'cmap': visual.infer_cmap(self.da),
                'cbar_kwargs': {
                    'label': f'{self.da.name} [{self.da.units}]',
                    'aspect': 10,
                },
            }
            _plt_kws = utils.update_dict(_plt_kws, kws)
            self.da.plot.contourf(ax=ax, **_plt_kws)

        else:
            # zonal mean, timeseries, others
            if figsize is None: figsize = (6, 3)
            if ax is None: fig, ax = plt.subplots(figsize=figsize)
            _plt_kws = {}
            _plt_kws = utils.update_dict(_plt_kws, kws)
            self.da.plot(ax=ax, **_plt_kws)
            ax.set_ylabel(f'{self.da.name} [{self.da.units}]')

        if title is None and 'long_name' in self.da.attrs:
            title = self.da.attrs['long_name']

        ax.set_title(title, weight='bold')

        if 'fig' in locals():
            return fig, ax
        else:
            return ax
    