#import all packages
from herbie import FastHerbie
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# ---- Configuration ----
DATA_FILE   = "/courses/meteo473/sp26/473_sp26_group7/data/gfs_event1.nc"
OUTPUT_DIR  = "/courses/meteo473/sp26/473_sp26_group7/website/images"
# ----
os.makedirs(OUTPUT_DIR, exist_ok=True)


#call in the variables
ds_a = xr.open_dataset("Marecmwf_20260323Mar.nc")
ds_b = xr.open_dataset("10fgMarecmwf_20260323Mar.nc")
ds2 = xr.merge([ds_a, ds_b], compat = 'override')
ds2

#useful variables
temp=ds2['t2m']
snow=ds2['sf']
dew=ds2['d2m']
wind_gust=ds2['fg10']
precip=ds2['tp']

#unit conversions
temp_f = (temp - 273.15) * 9/5 + 32
dew_f = (dew - 273.15) * 9/5 + 32
precip_mm = precip * 1000
wind_mph = wind_gust * 2.237
snow_mm = snow * 1000
times = ds2['valid_time'].values

#Relavtive Humidity Def: 
def calc_rh(T, Td):
    return 100 * np.exp((17.625 * Td)/(243.04 + Td)) / np.exp((17.625 * T)/(243.04 + T))

#Wet Bulb Def: 
def wet_bulb_temp(T, RH):
    return T * np.arctan(0.151977 * (RH + 8.313659)**0.5) + \
           np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
           0.00391838 * RH**1.5 * np.arctan(0.023101 * RH) - 4.686035

def wbgt_calc(T, Tw, Tg):
    # ALL inputs in Celsius
    return (0.7 * Tw) + (0.2 * Tg) + (0.1 * T)

# Store results
precip_risk = []
wind_risk = []
wbgt_risk = []
field_risk = []

times = ds2['valid_time'].values
#getting the valid time information
for t in range(len(times)):

    T = temp_f.isel(valid_time=t).values
    Td = dew_f.isel(valid_time=t).values
    P = precip_mm.isel(valid_time=t).values
    S = snow_mm.isel(valid_time=t).values
    W = wind_mph.isel(valid_time=t).values

    # conversions
    W_ms = W * 0.44704
    T_c = (T - 32) * 5/9
    Td_c = (Td - 32) * 5/9

    # ==============================
    # HUMIDITY + WET BULB
    # ==============================
    RH = calc_rh(T_c, Td_c)
    RH = np.clip(RH, 0, 100)

    Tw = wet_bulb_temp(T_c, RH)

    # ==============================
    # Tg (STABLE EMPIRICAL MODEL)
    # ==============================
    Tg = T_c + 1.5 + (0.15 * T_c) - (0.25 * W_ms)

    # ==============================
    # WBGT (FIXED)
    # ==============================
    WBGT = (wbgt_calc(T_c, Tw, Tg)*1.8)+32

temp_f = (ds2['t2m'].values - 273.15)*(9/5) + 32
snow= (ds2['sf'].values)*10 #10 inches of snow typically equals about 1 inch of water
wind_gust=(ds2['fg10'].values)*2.2369 #into mph
precip=(ds2['tp'].values)*39.37
dew_f=(ds2['d2m'].values - 273.15)*(9/5) + 32
lats = ds2['latitude'].values
lons = ds2['longitude'].values

temp_f_2d = temp_f[0, :, :] 
dew_f_2d=dew_f[0,:,:]
precip_2d=precip[1,:,:]
snow_2d=snow[0,:,:]
wind_gust_2d=wind_gust[1,:,:]

WBGT = (wbgt_calc(T_c, Tw, Tg)*1.8)+32


#Calling variables for next run
ds_a = xr.open_dataset("Marecmwf_20260323Mar.nc")
ds_b = xr.open_dataset("10fgMarecmwf_20260323Mar.nc")
ds2 = xr.merge([ds_a, ds_b], compat = 'override')
ds2


import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============================
# Lat/Lon
# ==============================
lon = ds2['longitude'].values
lat = ds2['latitude'].values
lon2d, lat2d = np.meshgrid(lon, lat)

# ==============================
# State College Coordinates
# ==============================
sc_lon = -77.8600
sc_lat = 40.7934

# ==============================
# RH FUNCTION
# ==============================
def calc_rh(temp_f, dew_f):
    temp_c = (temp_f - 32) * 5/9
    dew_c  = (dew_f - 32) * 5/9

    es = 6.112 * np.exp((17.67 * temp_c)/(temp_c + 243.5))
    e  = 6.112 * np.exp((17.67 * dew_c)/(dew_c + 243.5))

    rh = (e / es) * 100
    return np.clip(rh, 0, 100)

# ==============================
# COLORMAP
# ==============================
colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#8E44AD"]
risk_cmap = ListedColormap(colors)
bounds = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
norm = BoundaryNorm(bounds, risk_cmap.N)

# ==============================
# SELECT TIME STEPS
# ==============================
times_idx = [0, 2, 4, 6, 8, 10]
times_actual = ds2.valid_time.values[times_idx]

# ==============================
# STORAGE
# ==============================
PUI_list = []
temp_risk_list = []
wind_risk_list = []
precip_risk_list = []
WBGT_risk_list = []

# ==============================
# LOOP 
# ==============================
for i in times_idx:

    tempf2d = (ds2['t2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    dewf2d = (ds2['d2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    windgust2d = ds2['fg10'].isel(valid_time=i).values * 2.2369

    if i > 0:
        accum_precip = (
            ds2['tp'].isel(valid_time=i).values -
            ds2['tp'].isel(valid_time=i-1).values
        ) * 39.37
    else:
        accum_precip = ds2['tp'].isel(valid_time=i).values * 39.37

    RH = calc_rh(tempf2d, dewf2d)
    Tw = wet_bulb_temp(tempf2d, RH)
    WBGT = (wbgt_calc(T_c, Tw, Tg)*1.8)+32

    # --- Temperature Risk ---
    temp_risk = np.zeros_like(tempf2d)
    temp_risk[tempf2d < 40] = (40 - tempf2d[tempf2d < 40]) / 40
    mask = (tempf2d >= 40) & (tempf2d <= 75)
    temp_risk[mask] = 0
    mask = tempf2d > 75
    temp_risk[mask] = (tempf2d[mask] - 75) / 25
    temp_risk = np.clip(temp_risk, 0, 1)

    # --- WBGT Risk ---
    WBGT_risk = np.zeros_like(WBGT)
    WBGT_risk[(WBGT >= 80) & (WBGT < 85)] = 0.25
    WBGT_risk[(WBGT >= 85) & (WBGT < 88)] = 0.5
    WBGT_risk[(WBGT >= 88) & (WBGT < 90)] = 0.75
    WBGT_risk[WBGT >= 90] = 1.0

    # --- Wind Risk ---
    wind_risk = np.zeros_like(windgust2d)
    wind_risk[(windgust2d >= 15) & (windgust2d < 25)] = 0.3
    wind_risk[(windgust2d >= 25) & (windgust2d < 40)] = 0.6
    wind_risk[windgust2d >= 40] = 1.0

    # --- Precip Risk ---
    precip_risk = np.zeros_like(accum_precip)
    precip_risk[(accum_precip >= 0.05) & (accum_precip < 0.25)] = 0.3
    precip_risk[(accum_precip >= 0.25) & (accum_precip < 0.75)] = 0.6
    precip_risk[accum_precip >= 0.75] = 1.0

    temp_risk = np.nan_to_num(temp_risk)
    wind_risk = np.nan_to_num(wind_risk)
    precip_risk = np.nan_to_num(precip_risk)

    PUI = temp_risk + wind_risk + precip_risk + WBGT_risk
    PUI = np.nan_to_num(PUI)

    PUI_list.append(PUI)
    temp_risk_list.append(temp_risk)
    wind_risk_list.append(wind_risk)
    precip_risk_list.append(precip_risk)
    WBGT_risk_list.append(WBGT_risk)

# ==============================
# MAP FORMAT FUNCTION
# ==============================
def format_map(ax):
    ax.set_extent([-80.6, -74.5, 39.5, 42.5])
    ax.add_feature(cfeature.STATES, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # --- State College dot ---
    ax.scatter(
        sc_lon, sc_lat,
        color='red',
        s=80,
        edgecolor='black',
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    ax.text(
        sc_lon + 0.1, sc_lat + 0.1,
        "State College",
        fontsize=9,
        transform=ccrs.PlateCarree()
    )

# ==============================
# PLOTTING (NOW GEOGRAPHIC)
# ==============================
for t_idx, i in enumerate(times_idx):

    valid_time = str(times_actual[t_idx])[:16]

    fig, axs = plt.subplots(
        2, 3, figsize=(18,10),
        subplot_kw={'projection': ccrs.Mercator()}
    )

    axs = axs.flatten()

    im0 = axs[0].pcolormesh(lon2d, lat2d, PUI_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[0].set_title("Threat Index")
    format_map(axs[0])
    plt.colorbar(im0, ax=axs[0], shrink=0.9, orientation='horizontal', pad=0.1)

    im1 = axs[1].pcolormesh(lon2d, lat2d, temp_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[1].set_title("Temperature Risk")
    format_map(axs[1])
    plt.colorbar(im1, ax=axs[1], shrink=0.9, orientation='horizontal', pad=0.1)

    im2 = axs[2].pcolormesh(lon2d, lat2d, wind_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[2].set_title("Wind Risk")
    format_map(axs[2])
    plt.colorbar(im2, ax=axs[2], shrink=0.9, orientation='horizontal', pad=0.1)

    im3 = axs[3].pcolormesh(lon2d, lat2d, precip_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[3].set_title("Precip Risk")
    format_map(axs[3])
    plt.colorbar(im3, ax=axs[3], shrink=0.9, orientation='horizontal', pad=0.1)

    im4 = axs[4].pcolormesh(lon2d, lat2d, WBGT_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[4].set_title("WBGT Risk")
    format_map(axs[4])
    plt.colorbar(im4, ax=axs[4], shrink=0.9, orientation='horizontal', pad=0.1)

    axs[5].axis('off')

    init_time = pd.to_datetime(ds2['time'].values).tz_localize('UTC').tz_convert('US/Eastern')
    valid_time_et = pd.to_datetime(valid_time).tz_localize('UTC').tz_convert('US/Eastern')

    fig.suptitle(
    f"Pennsylvania Meteorological Risk Fields (ECMWF)\n"
    f"Init: {init_time.strftime('%Y-%m-%d %H:%M %Z')} | "
    f"Valid: {valid_time_et.strftime('%Y-%m-%d %H:%M %Z')}",
    fontsize=16
    )

    plt.tight_layout()
    plt.savefig('march.png')
    plt.close()

#Calling variables for next run
ds_a = xr.open_dataset("10fgJanBadecmwf_20260125Jan.nc")
ds_b = xr.open_dataset("JanBadecmwf_20260125Jan.nc")
ds2 = xr.merge([ds_a, ds_b], compat = 'override')
ds2

# ==============================
# SELECT TIME STEPS
# ==============================
times_idx = [0, 2, 4, 6, 8, 10]
times_actual = ds2.valid_time.values[times_idx]

# ==============================
# STORAGE
# ==============================
PUI_list = []
temp_risk_list = []
wind_risk_list = []
precip_risk_list = []
WBGT_risk_list = []

# ==============================
# LOOP (UNCHANGED LOGIC)
# ==============================
for i in times_idx:

    tempf2d = (ds2['t2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    dewf2d = (ds2['d2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    windgust2d = ds2['fg10'].isel(valid_time=i).values * 2.2369

    if i > 0:
        accum_precip = (
            ds2['tp'].isel(valid_time=i).values -
            ds2['tp'].isel(valid_time=i-1).values
        ) * 39.37
    else:
        accum_precip = ds2['tp'].isel(valid_time=i).values * 39.37

    RH = calc_rh(tempf2d, dewf2d)
    Tw = wet_bulb_temp(tempf2d, RH)
    WBGT = (wbgt_calc(T_c, Tw, Tg)*1.8)+32

    # --- Temperature Risk ---
    temp_risk = np.zeros_like(tempf2d)
    temp_risk[tempf2d < 40] = (40 - tempf2d[tempf2d < 40]) / 40
    mask = (tempf2d >= 40) & (tempf2d <= 75)
    temp_risk[mask] = 0
    mask = tempf2d > 75
    temp_risk[mask] = (tempf2d[mask] - 75) / 25
    temp_risk = np.clip(temp_risk, 0, 1)

    # --- WBGT Risk ---
    WBGT_risk = np.zeros_like(WBGT)
    WBGT_risk[(WBGT >= 80) & (WBGT < 85)] = 0.25
    WBGT_risk[(WBGT >= 85) & (WBGT < 88)] = 0.5
    WBGT_risk[(WBGT >= 88) & (WBGT < 90)] = 0.75
    WBGT_risk[WBGT >= 90] = 1.0

    # --- Wind Risk ---
    wind_risk = np.zeros_like(windgust2d)
    wind_risk[(windgust2d >= 15) & (windgust2d < 25)] = 0.3
    wind_risk[(windgust2d >= 25) & (windgust2d < 40)] = 0.6
    wind_risk[windgust2d >= 40] = 1.0

    # --- Precip Risk ---
    precip_risk = np.zeros_like(accum_precip)
    precip_risk[(accum_precip >= 0.05) & (accum_precip < 0.25)] = 0.3
    precip_risk[(accum_precip >= 0.25) & (accum_precip < 0.75)] = 0.6
    precip_risk[accum_precip >= 0.75] = 1.0

    temp_risk = np.nan_to_num(temp_risk)
    wind_risk = np.nan_to_num(wind_risk)
    precip_risk = np.nan_to_num(precip_risk)

    PUI = temp_risk + wind_risk + precip_risk + WBGT_risk
    PUI = np.nan_to_num(PUI)

    PUI_list.append(PUI)
    temp_risk_list.append(temp_risk)
    wind_risk_list.append(wind_risk)
    precip_risk_list.append(precip_risk)
    WBGT_risk_list.append(WBGT_risk)

# ==============================
# MAP FORMAT FUNCTION
# ==============================
def format_map(ax):
    ax.set_extent([-80.6, -74.5, 39.5, 42.5])
    ax.add_feature(cfeature.STATES, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # --- State College dot ---
    ax.scatter(
        sc_lon, sc_lat,
        color='red',
        s=80,
        edgecolor='black',
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    ax.text(
        sc_lon + 0.1, sc_lat + 0.1,
        "State College",
        fontsize=9,
        transform=ccrs.PlateCarree()
    )

# ==============================
# PLOTTING (NOW GEOGRAPHIC)
# ==============================
for t_idx, i in enumerate(times_idx):

    valid_time = str(times_actual[t_idx])[:16]

    fig, axs = plt.subplots(
        2, 3, figsize=(18,10),
        subplot_kw={'projection': ccrs.Mercator()}
    )

    axs = axs.flatten()

    im0 = axs[0].pcolormesh(lon2d, lat2d, PUI_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[0].set_title("Threat Index")
    format_map(axs[0])
    plt.colorbar(im0, ax=axs[0], orientation='horizontal', pad=0.1)

    im1 = axs[1].pcolormesh(lon2d, lat2d, temp_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[1].set_title("Temperature Risk")
    format_map(axs[1])
    plt.colorbar(im1, ax=axs[1], orientation='horizontal', pad=0.1)

    im2 = axs[2].pcolormesh(lon2d, lat2d, wind_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[2].set_title("Wind Risk")
    format_map(axs[2])
    plt.colorbar(im2, ax=axs[2], orientation='horizontal', pad=0.1)

    im3 = axs[3].pcolormesh(lon2d, lat2d, precip_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[3].set_title("Precip Risk")
    format_map(axs[3])
    plt.colorbar(im3, ax=axs[3], orientation='horizontal', pad=0.1)

    im4 = axs[4].pcolormesh(lon2d, lat2d, WBGT_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[4].set_title("WBGT Risk")
    format_map(axs[4])
    plt.colorbar(im4, ax=axs[4], orientation='horizontal', pad=0.1)

    axs[5].axis('off')

    init_time = pd.to_datetime(ds2['time'].values).tz_localize('UTC').tz_convert('US/Eastern')
    valid_time_et = pd.to_datetime(valid_time).tz_localize('UTC').tz_convert('US/Eastern')

    fig.suptitle(
    f"Pennsylvania Meteorological Risk Fields (ECMWF)\n"
    f"Init: {init_time.strftime('%Y-%m-%d %H:%M %Z')} | "
    f"Valid: {valid_time_et.strftime('%Y-%m-%d %H:%M %Z')}",
    fontsize=16
    )

    plt.tight_layout()
    plt.savefig('january.png')
    plt.close()

#calling variables for next run
ds_a = xr.open_dataset("AprCalmVarecmwf_20260406Apr.nc")
ds_b = xr.open_dataset("AprCalmecmwf_20260406Apr.nc")
ds2 = xr.merge([ds_a, ds_b], compat = 'override')

# ==============================
# SELECT TIME STEPS
# ==============================
times_idx = [0, 2, 4, 6, 8, 10]
times_actual = ds2.valid_time.values[times_idx]

# ==============================
# STORAGE
# ==============================
PUI_list = []
temp_risk_list = []
wind_risk_list = []
precip_risk_list = []
WBGT_risk_list = []

# ==============================
# LOOP (UNCHANGED LOGIC)
# ==============================
for i in times_idx:

    tempf2d = (ds2['t2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    dewf2d = (ds2['d2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    windgust2d = ds2['fg10'].isel(valid_time=i).values * 2.2369

    if i > 0:
        accum_precip = (
            ds2['tp'].isel(valid_time=i).values -
            ds2['tp'].isel(valid_time=i-1).values
        ) * 39.37
    else:
        accum_precip = ds2['tp'].isel(valid_time=i).values * 39.37

    RH = calc_rh(tempf2d, dewf2d)
    Tw = wet_bulb_temp(tempf2d, RH)
    WBGT = (wbgt_calc(T_c, Tw, Tg)*1.8)+32

    # --- Temperature Risk ---
    temp_risk = np.zeros_like(tempf2d)
    temp_risk[tempf2d < 40] = (40 - tempf2d[tempf2d < 40]) / 40
    mask = (tempf2d >= 40) & (tempf2d <= 75)
    temp_risk[mask] = 0
    mask = tempf2d > 75
    temp_risk[mask] = (tempf2d[mask] - 75) / 25
    temp_risk = np.clip(temp_risk, 0, 1)

    # --- WBGT Risk ---
    WBGT_risk = np.zeros_like(WBGT)
    WBGT_risk[(WBGT >= 80) & (WBGT < 85)] = 0.25
    WBGT_risk[(WBGT >= 85) & (WBGT < 88)] = 0.5
    WBGT_risk[(WBGT >= 88) & (WBGT < 90)] = 0.75
    WBGT_risk[WBGT >= 90] = 1.0

    # --- Wind Risk ---
    wind_risk = np.zeros_like(windgust2d)
    wind_risk[(windgust2d >= 15) & (windgust2d < 25)] = 0.3
    wind_risk[(windgust2d >= 25) & (windgust2d < 40)] = 0.6
    wind_risk[windgust2d >= 40] = 1.0

    # --- Precip Risk ---
    precip_risk = np.zeros_like(accum_precip)
    precip_risk[(accum_precip >= 0.05) & (accum_precip < 0.25)] = 0.3
    precip_risk[(accum_precip >= 0.25) & (accum_precip < 0.75)] = 0.6
    precip_risk[accum_precip >= 0.75] = 1.0

    temp_risk = np.nan_to_num(temp_risk)
    wind_risk = np.nan_to_num(wind_risk)
    precip_risk = np.nan_to_num(precip_risk)

    PUI = temp_risk + wind_risk + precip_risk + WBGT_risk
    PUI = np.nan_to_num(PUI)

    PUI_list.append(PUI)
    temp_risk_list.append(temp_risk)
    wind_risk_list.append(wind_risk)
    precip_risk_list.append(precip_risk)
    WBGT_risk_list.append(WBGT_risk)

# ==============================
# MAP FORMAT FUNCTION
# ==============================
def format_map(ax):
    ax.set_extent([-80.6, -74.5, 39.5, 42.5])
    ax.add_feature(cfeature.STATES, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # --- State College dot ---
    ax.scatter(
        sc_lon, sc_lat,
        color='red',
        s=80,
        edgecolor='black',
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    ax.text(
        sc_lon + 0.1, sc_lat + 0.1,
        "State College",
        fontsize=9,
        transform=ccrs.PlateCarree()
    )

# ==============================
# PLOTTING (NOW GEOGRAPHIC)
# ==============================
for t_idx, i in enumerate(times_idx):

    valid_time = str(times_actual[t_idx])[:16]

    fig, axs = plt.subplots(
        2, 3, figsize=(18,10),
        subplot_kw={'projection': ccrs.Mercator()}
    )

    axs = axs.flatten()

    im0 = axs[0].pcolormesh(lon2d, lat2d, PUI_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[0].set_title("Threat Index")
    format_map(axs[0])
    plt.colorbar(im0, ax=axs[0], orientation='horizontal', pad=0.1)

    im1 = axs[1].pcolormesh(lon2d, lat2d, temp_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[1].set_title("Temperature Risk")
    format_map(axs[1])
    plt.colorbar(im1, ax=axs[1], orientation='horizontal', pad=0.1)

    im2 = axs[2].pcolormesh(lon2d, lat2d, wind_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[2].set_title("Wind Risk")
    format_map(axs[2])
    plt.colorbar(im2, ax=axs[2], orientation='horizontal', pad=0.1)

    im3 = axs[3].pcolormesh(lon2d, lat2d, precip_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[3].set_title("Precip Risk")
    format_map(axs[3])
    plt.colorbar(im3, ax=axs[3], orientation='horizontal', pad=0.1)

    im4 = axs[4].pcolormesh(lon2d, lat2d, WBGT_risk_list[t_idx],
                           cmap=risk_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    axs[4].set_title("WBGT Risk")
    format_map(axs[4])
    plt.colorbar(im4, ax=axs[4], orientation='horizontal', pad=0.1)

    axs[5].axis('off')


    init_time = pd.to_datetime(ds2['time'].values).tz_localize('UTC').tz_convert('US/Eastern')
    valid_time_et = pd.to_datetime(valid_time).tz_localize('UTC').tz_convert('US/Eastern')

    fig.suptitle(
    f"Pennsylvania Meteorological Risk Fields (ECMWF)\n"
    f"Init: {init_time.strftime('%Y-%m-%d %H:%M %Z')} | "
    f"Valid: {valid_time_et.strftime('%Y-%m-%d %H:%M %Z')}",
    fontsize=16
    )

    plt.tight_layout()
    plt.savefig('april.png')
    plt.close()

#calling variables for threat index run
ds_a = xr.open_dataset("Marecmwf_20260323Mar.nc")
ds_b = xr.open_dataset("10fgMarecmwf_20260323Mar.nc")
ds2 = xr.merge([ds_a, ds_b], compat = 'override')
ds2

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm



# ==============================
# TIME STEPS
# ==============================
times_idx = list(range(0, 14, 1))
times_actual = ds2.valid_time.values[times_idx]

# ==============================
# STORAGE
# ==============================
PUI_list = []
temp_risk_list = []
wind_risk_list = []
precip_risk_list = []
WBGT_risk_list = []

# ==============================
# LOOP (UNCHANGED PHYSICS)
# ==============================
for i in times_idx:

    tempf2d = (ds2['t2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    dewf2d = (ds2['d2m'].isel(valid_time=i).values - 273.15)*(9/5)+32
    windgust2d = ds2['fg10'].isel(valid_time=i).values * 2.2369

    if i > 0:
        accum_precip = (
            ds2['tp'].isel(valid_time=i).values -
            ds2['tp'].isel(valid_time=i-1).values
        ) * 39.37
    else:
        accum_precip = ds2['tp'].isel(valid_time=i).values * 39.37

    RH = calc_rh(tempf2d, dewf2d)
    Tw = wet_bulb_temp(tempf2d, RH)
    WBGT = (wbgt_calc(T_c, Tw, Tg)*1.8)+32

    # --- Temperature Risk ---
    temp_risk = np.zeros_like(tempf2d)
    temp_risk[tempf2d < 40] = (40 - tempf2d[tempf2d < 40]) / 40
    mask = (tempf2d >= 40) & (tempf2d <= 75)
    temp_risk[mask] = 0
    mask = tempf2d > 75
    temp_risk[mask] = (tempf2d[mask] - 75) / 25
    temp_risk = np.clip(temp_risk, 0, 1)

    # --- WBGT Risk ---
    WBGT_risk = np.zeros_like(WBGT)
    WBGT_risk[(WBGT >= 80) & (WBGT < 85)] = 0.25
    WBGT_risk[(WBGT >= 85) & (WBGT < 88)] = 0.5
    WBGT_risk[(WBGT >= 88) & (WBGT < 90)] = 0.75
    WBGT_risk[WBGT >= 90] = 1.0

    # --- Wind Risk ---
    wind_risk = np.zeros_like(windgust2d)
    wind_risk[(windgust2d >= 15) & (windgust2d < 25)] = 0.3
    wind_risk[(windgust2d >= 25) & (windgust2d < 40)] = 0.6
    wind_risk[windgust2d >= 40] = 1.0

    # --- Precip Risk ---
    precip_risk = np.zeros_like(accum_precip)
    precip_risk[(accum_precip >= 0.05) & (accum_precip < 0.25)] = 0.3
    precip_risk[(accum_precip >= 0.25) & (accum_precip < 0.75)] = 0.6
    precip_risk[accum_precip >= 0.75] = 1.0

    temp_risk = np.nan_to_num(temp_risk)
    wind_risk = np.nan_to_num(wind_risk)
    precip_risk = np.nan_to_num(precip_risk)

    PUI = temp_risk + wind_risk + precip_risk + WBGT_risk
    PUI = np.nan_to_num(PUI)

    PUI_list.append(PUI)

# ==============================
# PLOTTING OVER PA MAP
# ==============================
for i in range(0, 14):

    fig, ax = plt.subplots(
        figsize=(9,7),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    fig.subplots_adjust(top=1.3)
    # --- MAP BASE ---
    ax.set_extent([-80.6, -74.5, 39.5, 42.5])
    ax.add_feature(cfeature.STATES, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # --- PLOT FIELD ---
    im = ax.pcolormesh(
        lon2d, lat2d,
        PUI_list[i],
        cmap=risk_cmap,
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    # --- STATE COLLEGE DOT ---
    ax.scatter(
        sc_lon, sc_lat,
        color='red',
        s=80,
        edgecolor='black',
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    ax.text(
        sc_lon + 0.1,
        sc_lat + 0.1,
        "State College",
        fontsize=9,
        transform=ccrs.PlateCarree()
    )

    # --- TITLE ---
    init_time = pd.to_datetime(ds2['time'].values).tz_localize('UTC').tz_convert('US/Eastern')
    valid_time_et = pd.to_datetime(valid_time).tz_localize('UTC').tz_convert('US/Eastern')

    fig.suptitle(
    f"Pennsylvania Meteorological Risk Fields (ECMWF)\n"
    f"Init: {init_time.strftime('%Y-%m-%d %H:%M %Z')} | "
    f"Valid: {valid_time_et.strftime('%Y-%m-%d %H:%M %Z')}",
    fontsize=16
    )
    

    # --- COLORBAR ---
    plt.colorbar(im, ax=ax, shrink=1,orientation='horizontal', pad=0.05)

    # ==============================
    # SAVE FIGURE
    # ==============================
    filename = f"pa_threat_maps/threat_{i:03d}.png"
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()

    print('Done.')