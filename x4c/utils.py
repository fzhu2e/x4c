import numpy as np
import xarray as xr
import xesmf as xe
import colorama as ca
import datetime
import collections.abc
import cartopy.util

def p_header(text):
    print(ca.Fore.CYAN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_hint(text):
    print(ca.Fore.LIGHTBLACK_EX + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_success(text):
    print(ca.Fore.GREEN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_fail(text):
    print(ca.Fore.RED + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_warning(text):
    print(ca.Fore.YELLOW + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def regrid_cam_se(ds, weight_file):
    """
    Regrid CAM-SE output using an existing ESMF weights file.

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset to be regridded. Must have the `ncol` dimension.
    weight_file: str or Path
        Path to existing ESMF weights file

    Returns
    -------
    regridded
        xarray.Dataset after regridding.
    """
    dataset = ds.copy()
    assert isinstance(dataset, xr.Dataset)
    weights = xr.open_dataset(weight_file)

    # input variable shape
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shapew
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    print(f"Regridding from {in_shape} to {out_shape}")

    # Insert dummy dimension
    vars_with_ncol = [name for name in dataset.variables if "ncol" in dataset[name].dims]
    updated = dataset.copy().update(
        dataset[vars_with_ncol].transpose(..., "ncol").expand_dims("dummy", axis=-2)
    )

    # construct a regridder
    # use empty variables to tell xesmf the right shape
    # https://github.com/pangeo-data/xESMF/issues/202
    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )

    regridder = xe.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        method="test",
        reuse_weights=True,
        periodic=True,
    )

    # Actually regrid, after renaming
    regridded = regridder(updated.rename({"dummy": "lat", "ncol": "lon"}), keep_attrs=True)

    # merge back any variables that didn't have the ncol dimension
    # And so were not regridded
    return xr.merge([dataset.drop_vars(regridded.variables), regridded])

def annualize(ds, months=None):
    months = list(range(1, 13)) if months is None else np.abs(months)
    sds = ds.sel(time=ds['time.month'].isin(months))
    anchor = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    idx = months[-1]-1
    ds_ann = sds.resample(time=f'YE-{anchor[idx]}').mean()
    return ds_ann

def monthly2annual(ds):
    month_length = ds.time.dt.days_in_month
    wgts_mon = month_length.groupby('time.year') / month_length.groupby('time.year').mean()
    ds_ann = (ds * wgts_mon).groupby('time.year').mean('time')
    return ds_ann.rename({'year':'time'})

def monthly2season(ds):
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby('time.season') / month_length.groupby('time.season').mean()
    ds_season = (ds * wgts).groupby('time.season').mean('time')
    return ds_season

def geo_mean(da, lat_min=-90, lat_max=90, lon_min=0, lon_max=360, lat_name='lat', lon_name='lon'):
    ''' Calculate the geographical mean value of the climate field.

    Args:
        lat_min (float): the lower bound of latitude for the calculation.
        lat_max (float): the upper bound of latitude for the calculation.
        lon_min (float): the lower bound of longitude for the calculation.
        lon_max (float): the upper bound of longitude for the calculation.
    '''
    # calculation
    mask_lat = (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max)
    mask_lon = (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max)

    dac = da.sel(
        {
            lat_name: da[lat_name][mask_lat],
            lon_name: da[lon_name][mask_lon],
        }
    )

    wgts = np.cos(np.deg2rad(dac[lat_name]))
    m = dac.weighted(wgts).mean((lon_name, lat_name))
    return m

def update_attrs(da, da_src):
    da.attrs = dict(da_src.attrs)
    if 'comp' in da.attrs and 'time' in da.coords:
        da.time.attrs['long_name'] = 'Model Year'

    return da

def update_ds(ds, path, vn=None, comp=None, grid=None, adjust_month=False,
              gw_name=None, lat_name=None, lon_name=None):
    if adjust_month:
        ds['time'] = ds['time'].get_index('time') - datetime.timedelta(days=1)

    ds.attrs['path'] = path
    if vn is not None: ds.attrs['vn'] = vn
    if comp is not None: ds.attrs['comp'] = comp
    if grid is not None: ds.attrs['grid'] = grid

    if 'comp' in ds.attrs:
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

        gw_name = grid_weight_dict[comp] if gw_name is None else gw_name
        lat_name = lat_dict[comp] if lat_name is None else lat_name
        lon_name = lon_dict[comp] if lon_name is None else lon_name

    if gw_name is not None: ds['gw'] = ds[gw_name]
    if lat_name is not None: ds['lat'] = ds[lat_name]
    if lon_name is not None: ds['lon'] = ds[lon_name]

    return ds

def infer_months_char(months):
    char_list = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    out_str = ''
    for i in months:
        out_str += char_list[np.abs(i)-1]
    return out_str


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def add_cyclic_point(da):
    data_wrap, lon_wrap = cartopy.util.add_cyclic_point(da.values, coord=da.lon)
    da_new_coords = da.coords.copy()
    da_new_coords['lon'] = lon_wrap
    da_new_coords
    da_wrap = xr.DataArray(data_wrap, coords=da_new_coords)
    da_wrap.attrs = da.attrs.copy()
    return da_wrap

def ann_modifier(da, ann_method, long_name=None):
    if 'long_name' in da.attrs:
        long_name = da.attrs['long_name'] if long_name is None else long_name
    else:
        long_name = da.name

    if ann_method == 'ann':
        da_out = da.x.annualize()
        da_out.attrs['long_name'] = f'{long_name} (Annual)'
    else:
        months = [int(s) for s in ann_method.split(',')]
        months_char = infer_months_char(months)
        da_out = da.x.annualize(months=months)
        da_out.attrs['long_name'] = f'{long_name} ({months_char})'

    return da_out

def convert_units(da, units=None):
    if units is not None:
        if 'units' in da.attrs:
            if da.attrs['units'] == 'K' and units == 'degC':
                da -= 273.15
                da.attrs['units'] = '°C'
            elif da.attrs['units'] == 'degC' and units == 'K':
                da += 273.15
                da.attrs['units'] = 'K'
            elif da.attrs['units'] == 'degC' and units == 'degC' or units is None:
                da.attrs['units'] = '°C'
        else:
            p_warning("The inpu `xarray.DataArray` doesn't have a unit.")

    return da