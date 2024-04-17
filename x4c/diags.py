from . import utils
import xarray as xr
import numpy as np

class DiagCalc:
    def calc_ts_GMST(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        vn = 'TS'
        case.load(vn, load_idx=load_idx, adjust_month=adjust_month)
        da_degC = case.ds[vn] - 273.15
        da_degC.attrs['units'] = '°C'

        if mean_method == 'ann':
            da = da_degC.x.annualize().x.gm
            da.attrs['long_name'] = 'Global Mean Surface Temperature (Annual)'
        else:
            months = [int(s) for s in mean_method.split(',')]
            months_char = utils.infer_months_char(months)
            da = da_degC.x.annualize(months=months).x.gm
            da.attrs['long_name'] = f'Global Mean Surface Temperature ({months_char})'
        return da

    def calc_map_TS(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        vn = 'TS'
        case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
        da_degC = case.ds[vn] - 273.15
        da_degC.attrs['units'] = '°C'

        if mean_method == 'ann':
            da = da_degC.x.annualize().mean('time')
            da.attrs['long_name'] = 'Surface Temperature (Annual)'
        else:
            months = [int(s) for s in mean_method.split(',')]
            months_char = utils.infer_months_char(months)
            da = da_degC.x.annualize(months=months).mean('time')
            da.attrs['long_name'] = f'Surface Temperature ({months_char})'
        return da

    def calc_map_LST(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        vn = 'TS'
        case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
        da_degC = case.ds[vn] - 273.15
        da_degC.attrs['units'] = '°C'

        case.load('LANDFRAC', load_idx=load_idx, adjust_month=adjust_month, regrid=True)
        landfrac = case.ds['LANDFRAC'].x.annualize().mean('time')

        if mean_method == 'ann':
            da = da_degC.x.annualize().where(landfrac>0.5).mean('time')
            da.attrs['long_name'] = 'Land Surface Temperature (Annual)'
        else:
            months = [int(s) for s in mean_method.split(',')]
            months_char = utils.infer_months_char(months)
            da = da_degC.x.annualize(months=months).where(landfrac>0.5).mean('time')
            da.attrs['long_name'] = f'Land Surface Temperature ({months_char})'

        da.name = 'LST'
        return da

    def calc_map_SST(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        if 'SST' in case.vars_info:
            vn = 'SST'
            case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
            case.ds[vn].attrs['units'] = '°C'
        else:
            vn = 'TEMP'
            case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
            case.ds[vn].attrs['units'] = '°C'
            sst = case.ds.x[vn].isel(z_t=0)

        if mean_method == 'ann':
            da = sst.x.annualize().mean('time')
            da.attrs['long_name'] = 'Sea Surface Temperature (Annual)'
        else:
            months = [int(s) for s in mean_method.split(',')]
            months_char = utils.infer_months_char(months)
            da = sst.x.annualize(months=months).mean('time')
            da.attrs['long_name'] = f'Sea Surface Temperature ({months_char})'

        da.name = 'SST'
        return da

    def calc_map_MLD(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        vn = 'XMXL'
        case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
        da_tmp = case.ds[vn] / 100

        if mean_method == 'ann':
            da = da_tmp.x.annualize().mean('time')
            da.attrs['long_name'] = 'Mixed Layer Depth (Annual)'
        else:
            months = [int(s) for s in mean_method.split(',')]
            months_char = utils.infer_months_char(months)
            da = da_tmp.x.annualize(months=months).mean('time')
            da.attrs['long_name'] = f'Mixed Layer Depth ({months_char})'

        da.name = 'MLD'
        da.attrs['units'] = 'm'
        return da

    def calc_zm_LST(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        da = DiagCalc.calc_map_LST(case, load_idx=load_idx, adjust_month=adjust_month, mean_method=mean_method)
        da_zm = da.x.zm
        da_zm.attrs['long_name'] = f'Zonal Mean {da.attrs["long_name"]}'
        return da_zm

    def calc_zm_SST(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        da = DiagCalc.calc_map_SST(case, load_idx=load_idx, adjust_month=adjust_month, mean_method=mean_method)
        da_zm = da.x.zm
        da_zm.attrs['long_name'] = f'Zonal Mean {da.attrs["long_name"]}'
        return da_zm

    def calc_3d_MOC(case, load_idx=-1, adjust_month=True, mean_method='ann'):
        vn = 'MOC'
        case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=False)
        if mean_method == 'ann':
            da = case.ds[vn].x.annualize()
            da.attrs['long_name'] = 'Meridional Ocean Circulation (Annual)'
        else:
            months = [int(s) for s in mean_method.split(',')]
            months_char = utils.infer_months_char(months)
            da = case.ds[vn].x.annualize(months=months)
            da.attrs['long_name'] = f'Meridional Ocean Circulation ({months_char})'

        da_out = da.copy()
        da_out['moc_z'] = da['moc_z'] / 1e5  # unit: cm -> km
        da_out['moc_z'].attrs['units'] = 'km'
        return da_out

    def calc_ts_MOC(case, load_idx=-1, adjust_month=True, mean_method='ann', transport_reg=0, moc_z=slice(0.5, None), lat_aux_grid=slice(-90, -28)):
        da = DiagCalc.calc_3d_MOC(case, load_idx=load_idx, adjust_month=adjust_month, mean_method=mean_method)
        da_out = da.isel(transport_reg=transport_reg, moc_comp=0).sel(moc_z=moc_z, lat_aux_grid=lat_aux_grid).min(('moc_z', 'lat_aux_grid'))
        return da_out

    def calc_yz_MOC(case, load_idx=-1, adjust_month=True, mean_method='ann', transport_reg=0):
        da = DiagCalc.calc_3d_MOC(case, load_idx=load_idx, adjust_month=adjust_month, mean_method=mean_method)
        da_out = da.isel(transport_reg=transport_reg, moc_comp=0).mean('time')
        return da_out

class DiagPlot:
    kws_ts = {}
    kws_map = {}
    kws_zm = {}
    kws_yz = {}

    # ==========
    #  kws_ts
    # ----------
    kws_ts['GMST'] = {'ylim': [20, 30]}

    # ==========
    #  kws_map
    # ----------
    kws_map['TS'] = {'levels': np.linspace(0, 40, 21), 'cbar_kwargs': {'ticks': np.linspace(0, 40, 5)}}
    kws_map['LST'] = {'levels': np.linspace(0, 40, 21), 'cbar_kwargs': {'ticks': np.linspace(0, 40, 5)}}
    kws_map['SST'] = {'levels': np.linspace(0, 40, 21), 'cbar_kwargs': {'ticks': np.linspace(0, 40, 5)}}

    kws_map['MLD'] = {
        'levels': np.linspace(0, 500, 21),
        'cbar_kwargs': {'ticks': np.linspace(0, 500, 11)},
        'extend': 'max',
        'central_longitude': -30,
    }

    # ==========
    #  kws_zm
    # ----------
    kws_zm['LST'] = {'ylim': (-40, 40)}

    # ==========
    #  kws_yz
    # ----------
    kws_yz['MOC'] = {'levels': np.linspace(-20, 20, 21), 'cbar_kwargs': {'ticks': np.linspace(-20, 20, 5)}}


    # base function for timeseries (ts) plotting
    def plot_ts(case, diag_name, mean_method='ann', **kws):
        _kws = DiagPlot.kws_ts[diag_name].copy() if diag_name in DiagPlot.kws_ts else {}
        _kws = utils.update_dict(_kws, kws)
        fig_ax =  case.diags[f'ts:{diag_name}:{mean_method}'].x.plot(**_kws)

        return fig_ax

    # base function for map plotting
    def plot_map(case, diag_name, mean_method='ann', cyclic=False, **kws):
        _kws = DiagPlot.kws_map[diag_name].copy() if diag_name in DiagPlot.kws_map else {}
        _kws = utils.update_dict(_kws, kws)
        da = case.diags[f'map:{diag_name}:{mean_method}']
        if cyclic: da = utils.add_cyclic_point(da)

        if 'SSH' in case.vars_info:
            case.load('SSH', regrid=True)
            da_ssv = case.ds['SSH'].mean('time')
            if cyclic: da_ssv = utils.add_cyclic_point(da_ssv)
            fig_ax =  da.x.plot(ssv=da_ssv, **_kws)
        else:
            fig_ax =  da.x.plot(**_kws)

        return fig_ax

    # base function for vertical slice (yz) plotting
    def plot_yz(case, diag_name, mean_method='ann', **kws):
        _kws = DiagPlot.kws_yz[diag_name].copy() if diag_name in DiagPlot.kws_yz else {}
        _kws = utils.update_dict(_kws, kws)

        fig_ax =  case.diags[f'yz:{diag_name}:{mean_method}'].x.plot(**_kws)
        ax = fig_ax[-1] if isinstance(fig_ax, tuple) else fig_ax

        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xticklabels(['90°S', '60°S', '30°S', 'EQ', '30°N', '60°N', '90°N'])
        ax.set_xlim([-90, 90])
        ax.set_xlabel('Latitude')

        ax.invert_yaxis()
        ax.set_yticks([0, 2, 4])
        ax.set_ylabel('Depth [km]')
        return fig_ax

    # base function for zonal mean (zm) plotting
    def plot_zm(case, diag_name, mean_method='ann', **kws):
        _kws = DiagPlot.kws_zm[diag_name].copy() if diag_name in DiagPlot.kws_zm else {}
        _kws = utils.update_dict(_kws, kws)

        fig_ax = case.diags[f'zm:{diag_name}:{mean_method}'].x.plot(**_kws)
        ax = fig_ax[-1] if isinstance(fig_ax, tuple) else fig_ax

        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xticklabels(['90°S', '60°S', '30°S', 'EQ', '30°N', '60°N', '90°N'])
        ax.set_xlim([-90, 90])
        ax.set_xlabel('Latitude')

        return fig_ax