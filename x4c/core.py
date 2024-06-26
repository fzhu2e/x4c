import xarray as xr
xr.set_options(keep_attrs=True)

import copy
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geocat.comp as gc


from . import utils, visual
import os
dirpath = os.path.dirname(__file__)

def load_dataset(path, adjust_month=False, comp=None, grid=None, vn=None, **kws):
    ''' Load a netCDF file and form a `xarray.Dataset`

    Args:
        path (str): path to the netCDF file
        adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
        comp (str): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    ds = xr.load_dataset(path, **kws)
    ds = utils.update_ds(ds, vn=vn, path=path, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

def open_dataset(path, adjust_month=False, comp=None, grid=None, vn=None, **kws):
    ''' Open a netCDF file and form a `xarray.Dataset` with a lazy load mode

    Args:
        path (str): path to the netCDF file
        adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
        comp (str): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    ds = xr.open_dataset(path, **kws)
    ds = utils.update_ds(ds, vn=vn, path=path, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

def open_mfdataset(paths, adjust_month=False, comp=None, grid=None, vn=None, **kws):
    ''' Open multiple netCDF files and form a `xarray.Dataset` in a lazy load mode

    Args:
        path (str): path to the netCDF file
        adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
        comp (str): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    ds = xr.open_mfdataset(paths, **kws)
    ds = utils.update_ds(ds, vn=vn, path=paths, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

@xr.register_dataset_accessor('x')
class XDataset:
    def __init__(self, ds=None):
        self.ds = ds

    def regrid(self, dlon=1, dlat=1, weight_file=None, gs='T', method='bilinear', periodic=True):
        ''' Regrid the CESM output to a normal lat/lon grid

        Args:
            dlon (float): longitude spacing
            dlat (float): latitude spacing
            weight_file (str): the path to an ESMF-generated weighting file for regridding
            gs (str): grid style in 'T' or 'U' for ocean grid
            method (str): regridding method for ocean grid
            periodic (bool): the assumption of the periodicity of the data when perform the regrid method

        '''
        comp = self.ds.attrs['comp']
        grid = self.ds.attrs['grid']

        if grid in ['ne16', 'ne30', 'ne120']:
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
        # utils.p_success(f'Dataset regridded to regular grid: [dlon: {dlon} x dlat: {dlat}]')
        if 'lat' in ds_rgd.attrs: del(ds_rgd.attrs['lat'])
        if 'lon' in ds_rgd.attrs: del(ds_rgd.attrs['lon'])
        return ds_rgd

    def get_plev(self, ps, vn=None, lev_mode='hybrid', **kws):
        _kws = {'lev_dim': 'lev'}
        _kws.update(kws)
        if vn is None:
            da = self.da
            vn = self.ds.attrs['vn']
        else:
            da = self.ds[vn]

        if isinstance(ps, xr.Dataset):
            ps_da = ps['PS']
        elif isinstance(ps, xr.DataArray):
            ps_da = ps

        if lev_mode == 'hybrid':
            da_plev = gc.interpolation.interp_hybrid_to_pressure(da, ps_da, self.ds['hyam'], self.ds['hybm'], **_kws)
        else:
            raise ValueError('`lev_mode` unknown')

        ds_plev = self.ds.copy()
        del(ds_plev[vn])
        ds_plev[vn] = da_plev
        return ds_plev


    def annualize(self, months=None):
        ''' Annualize/seasonalize a `xarray.Dataset`

        Args:
            months (list of int): a list of integers to represent month combinations,
                e.g., `None` means calendar year annualization, [7,8,9] means JJA annualization, and [-12,1,2] means DJF annualization

        '''
        ds_ann = utils.annualize(self.ds, months=months)
        ds_ann.attrs = dict(self.ds.attrs)
        return ds_ann


    def __getitem__(self, key):
        da = self.ds[key]

        if 'path' in self.ds.attrs:
            da.attrs['path'] = self.ds.attrs['path']

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

    @property
    def da(self):
        ''' get its `xarray.DataArray` version '''
        if 'vn' in self.ds.attrs:
            vn = self.ds.attrs['vn']
            return self.ds.x[vn]
        else:
            raise ValueError('`vn` not existed in `Dataset.attrs`')

    @property
    def climo(self):
        ds = self.ds.groupby('time.month').mean(dim='time')
        ds.attrs['climo_period'] = (self.ds['time.year'].values[0], self.ds['time.year'].values[-1])
        if 'comp' in self.ds.attrs: ds.attrs['comp'] = self.ds.attrs['comp']
        if 'grid' in self.ds.attrs: ds.attrs['grid'] = self.ds.attrs['grid']
        if 'month' in ds.coords:
            ds = ds.rename({'month': 'time'})
        return ds

    def to_netcdf(self, path, **kws):
        for v in ['gw', 'lat', 'lon']:
            if v in self.ds.attrs: del(self.ds.attrs[v])

        return self.ds.to_netcdf(path, **kws)
        

@xr.register_dataarray_accessor('x')
class XDataArray:
    def __init__(self, da=None):
        self.da = da

    def annualize(self, months=None):
        ''' Annualize/seasonalize a `xarray.DataArray`

        Args:
            months (list of int): a list of integers to represent month combinations,
                e.g., [7,8,9] means JJA annualization, and [-12,1,2] means DJF annualization

        '''
        da = utils.annualize(self.da, months=months)
        da = utils.update_attrs(da, self.da)
        return da

    def regrid(self, **kws):
        ds_rgd = self.ds.x.regrid(**kws)
        da = ds_rgd.x.da
        da.name = self.da.name
        if 'lat' in da.attrs: del(da.attrs['lat'])
        if 'lon' in da.attrs: del(da.attrs['lon'])
        return da

    def to_netcdf(self, path, **kws):
        for v in ['gw', 'lat', 'lon']:
            if v in self.da.attrs: del(self.da.attrs[v])

        return self.da.to_netcdf(path, **kws)

    @property
    def ds(self):
        ''' get its `xarray.Dataset` version '''
        ds_tmp = self.da.to_dataset()

        for v in ['gw', 'lat', 'lon']:
            if v in self.da.attrs: ds_tmp[v] = self.da.attrs[v]

        for v in ['comp', 'grid']:
            if v in self.da.attrs: ds_tmp.attrs[v] = self.da.attrs[v]
        
        ds_tmp[self.da.name] = self.da
        ds_tmp.attrs['vn'] = self.da.name
        return ds_tmp

    @property
    def gm(self):
        ''' the global area-weighted mean '''
        gw = self.da.attrs['gw']
        da = self.da.weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Global Mean {da.attrs["long_name"]}'
        return da

    @property
    def nhm(self):
        ''' the NH area-weighted mean '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat>0).weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'NH Mean {da.attrs["long_name"]}'
        return da

    @property
    def shm(self):
        ''' the SH area-weighted mean '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat<0).weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'SH Mean {da.attrs["long_name"]}'
        return da

    @property
    def zm(self):
        ''' the zonal mean
        '''
        if 'lon' not in self.da.dims:
            da = self.da.x.regrid().mean('lon')
        else:
            da = self.da.mean('lon')

        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Zonal Mean {da.attrs["long_name"]}'
        return da

    @property
    def climo(self):
        da = self.da.groupby('time.month').mean(dim='time')
        da.attrs['climo_period'] = (self.da['time.year'].values[0], self.da['time.year'].values[-1])
        if 'comp' in self.da.attrs: da.attrs['comp'] = self.da.attrs['comp']
        if 'grid' in self.da.attrs: da.attrs['grid'] = self.da.attrs['grid']
        if 'month' in da.coords:
            da = da.rename({'month': 'time'})
        return da

    def geo_mean(self, ind=None, latlon_range=(-90, 90, 0, 360), **kws):
        ''' The lat-weighted mean given a lat/lon range or a climate index name

        Args:
            latlon_range (tuple or list): the lat/lon range for lat-weighted average 
                in format of (lat_min, lat_max, lon_min, lon_max)

            ind (str): a climate index name; supported names include:
            
                * 'nino3.4'
                * 'nino1+2'
                * 'nino3'
                * 'nino4'
                * 'tpi'
                * 'wp'
                * 'dmi'
                * 'iobw'
        '''

        if ind is None:
            lat_min, lat_max, lon_min, lon_max = latlon_range
            da = utils.geo_mean(self.da, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, **kws)
        elif ind == 'nino3.4':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(-170, 360), lon_max=np.mod(-120, 360), **kws)
        elif ind == 'nino1+2':
            da = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(-90, 360), lon_max=np.mod(-80, 360), **kws)
        elif ind == 'nino3':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(-150, 360), lon_max=np.mod(-90, 360), **kws)
        elif ind == 'nino4':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(160, 360), lon_max=np.mod(-150, 360), **kws)
        elif ind == 'wpi':
            # Western Pacific Index
            da = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(120, 360), lon_max=np.mod(150, 360), **kws)
        elif ind == 'tpi':
            # Tri-Pole Index
            v1 = utils.geo_mean(self.da, lat_min=25, lat_max=45, lon_min=np.mod(140, 360), lon_max=np.mod(-145, 360), **kws)
            v2 = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(170, 360), lon_max=np.mod(-90, 360), **kws)
            v3 = utils.geo_mean(self.da, lat_min=-50, lat_max=-15, lon_min=np.mod(150, 360), lon_max=np.mod(-160, 360), **kws)
            da = v2 - (v1 + v3)/2
        elif ind == 'dmi':
            # Indian Ocean Dipole Mode
            dmiw = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=50 ,lon_max=70, **kws)
            dmie = utils.geo_mean(self.da,lat_min=-10,lat_max=0,lon_min=90,lon_max=110, **kws)
            da = dmiw - dmie
        elif ind == 'iobw':
            # Indian Ocean Basin Wide
            da =  utils.geo_mean(self.da, lat_min=-20, lat_max=20, lon_min=40 ,lon_max=100, **kws)
        else:
            raise ValueError('`ind` options: {"nino3.4", "nino1+2", "nino3", "nino4", "wpi", "tpi", "dmi", "iobw"}')

        da.attrs = dict(self.da.attrs)
        if 'comp' in da.attrs and 'time' in da.coords:
            da.time.attrs['long_name'] = 'Model Year'
        return da

    def plot(self, title=None, figsize=None, ax=None, latlon_range=None,
             projection='Robinson', transform='PlateCarree', central_longitude=180, proj_args=None,
             add_gridlines=False, gridline_labels=True, gridline_style='--', ssv=None,
             coastline_zorder=99, coastline_width=1, site_markersizes=100, df_sites=None, colname_dict=None, **kws):
        ''' The plotting functionality

        Args:
            title (str): figure title
            figsize (tuple or list): figure size in format of (w, h)
            ax (`matplotlib.axes`): a `matplotlib.axes`
            latlon_range (tuple or list): lat/lon range in format of (lat_min, lat_max, lon_min, lon_max)
            projection (str): a projection name supported by `Cartopy`
            transform (str): a projection name supported by `Cartopy`
            central_longitude (float): the central longitude of the map to plot
            proj_args (dict): other keyword arguments for projection
            add_gridlines (bool): if True, the map will be added with gridlines
            gridline_labels (bool): if True, the lat/lon ticklabels will appear
            gridline_style (str): the gridline style, e.g., '-', '--'
            ssv (`xarray.DataArray`): a sea surface variable used for plotting the coastlines
            coastline_zorder (int): the layer order for the coastlines
            coastline_width (float): the width of the coastlines
            df_sites (`pandas.DataFrame`): a `pandas.DataFrame` that stores the information of a collection of sites
            colname_dict (dict): a dictionary of column names for `df_sites` in the "key:value" format "assumed name:real name"

        '''
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

            if 'units' in self.da.attrs:
                cbar_lb = f'{self.da.name} [{self.da.units}]'
            else:
                cbar_lb = self.da.name

            _plt_kws = {
                'transform': _transform,
                'extend': 'both',
                'cmap': visual.infer_cmap(self.da),
                'cbar_kwargs': {
                    'label': cbar_lb,
                    'aspect': 10,
                },
            }
            _plt_kws = utils.update_dict(_plt_kws, kws)
            if 'add_colorbar' in kws and kws['add_colorbar'] is False:
                del(_plt_kws['cbar_kwargs'])

            if latlon_range is not None:
                lat_min, lat_max, lon_min, lon_max = latlon_range
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=_transform)

            if add_gridlines:
                gl = ax.gridlines(linestyle=gridline_style, draw_labels=gridline_labels)
                gl.top_labels = False
                gl.right_labels = False

            # add coastlines
            if ssv is not None:
                # using a sea surface variable with NaNs for coastline plotting
                ax.contour(ssv.lon, ssv.lat, np.isnan(ssv), levels=[0, 1], colors='k', transform=_transform, zorder=coastline_zorder, linewidths=coastline_width)
            elif 'comp' in self.da.attrs and (self.da.attrs['comp'] in ['ocn', 'ice']):
                # using NaNs in the dataarray itself for coastline plotting
                ax.contour(self.da.lon, self.da.lat, np.isnan(self.da), levels=[0, 1], colors='k', transform=_transform, zorder=coastline_zorder, linewidths=coastline_width)
            else:
                ax.coastlines(zorder=coastline_zorder, linewidth=coastline_width)

            im = self.da.plot.contourf(ax=ax, **_plt_kws)

            if df_sites is not None:
                colname_dict = {} if colname_dict is None else colname_dict
                _colname_dict={'lat': 'lat', 'lon':'lon', 'value': 'value', 'type': 'type'}
                _colname_dict.update(colname_dict)
                site_lons = df_sites[_colname_dict['lon']] if _colname_dict['lon'] in df_sites else None
                site_lats = df_sites[_colname_dict['lat']] if _colname_dict['lat'] in df_sites else None
                site_vals = df_sites[_colname_dict['value']] if _colname_dict['value'] in df_sites else None
                site_types = df_sites[_colname_dict['type']] if _colname_dict['type'] in df_sites else None

                if site_types is None:
                    site_markers = 'o'

                    if site_vals is None:
                        site_colors = 'gray'
                    else:
                        site_colors = site_vals
                else:
                    site_markers = [visual.marker_dict[t] for t in site_types]

                cmap_obj = plt.get_cmap(_plt_kws['cmap'])
                norm = BoundaryNorm(im.levels, ncolors=cmap_obj.N, clip=True)
                ax.scatter(site_lons, site_lats, s=site_markersizes, marker=site_markers, edgecolors='k', c=site_colors,
                           zorder=99, transform=_transform, cmap=cmap_obj, norm=norm)

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
    