#!/usr/bin/env python3
import xarray as xr
import functions as f
import numpy as np
import climpred
from xclim import sdba
from glob import glob
import preprocessUtils as putils
import os
import random
import masks
import pandas as pd

def load_mask(region_name):
    mask = masks.load_mask(region_name)
    #Mask with np.nan for non-CONUS land values
    mask_anom = mask[putils.xarray_varname(mask)][0,:,:].values
    return(mask,mask_anom)


def return_short_array(path,test_start,test_end):
    y1 = str(pd.to_datetime(test_start).year)
    y2 = str(pd.to_datetime(test_end).year)

    outpaths = []
    for p in sorted(glob(path)):
        if (y1 in p) or (y2 in p):
            outpaths.append(p)
    return(outpaths)

def open_obs_and_baseline_files(region_name, week_lead, day_num, start_, end_, mask_anom, test_start, test_end):

    print(f'Loading template data from {test_start} to {test_end}')
    mask,mask_anom = load_mask(region_name)
    
    obs_anomaly_SubX_format_testing =xr.open_mfdataset(return_short_array(f'Data/GLEAM/RZSM_anomaly_reformat_SubX_format/{region_name}/RZSM_anomaly*',test_start,test_end)).sel(L=[day_num]).load()
    template_testing_only = obs_anomaly_SubX_format_testing.copy(deep=True)

    assert pd.to_datetime(start_).year == pd.to_datetime(end_).year, 'For this code to work properly, you must have the beginning and ending of the case study in the same year!'

    print(f'Loading observation, gefsv12 raw reforecast, and ecmwf raw reforecast during case study dates of {start_} to {end_}')
    obs_anomaly_SubX_format =xr.open_mfdataset(f'Data/GLEAM/RZSM_anomaly_reformat_SubX_format/{region_name}/RZSM_anomaly*{pd.to_datetime(start_).year}*.nc4').sel(L=[day_num]).sel(S=slice(start_,end_)).load()

    var_OUT = np.empty(shape=(obs_anomaly_SubX_format.Y.shape[0], obs_anomaly_SubX_format.X.shape[0])) #48x96
    
    #Mask the final output to be np.nan for ocean values
    var_OUT = np.where(mask_anom==1, np.nan, var_OUT)
    var_OUT[:,:] = 0
    
    #######################################   Reforecast baseline files   ###########################################################################
    # baseline_anomaly_file_list = sorted(glob('Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/RZSM*.nc'))
    if region_name =='CONUS':
        baseline_anomaly_file_list = sorted(glob(f'Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
    
    else:
        baseline_anomaly_file_list = sorted(glob(f'Data_{region_name}/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
        baseline_anomaly = xr.where(np.isnan(mask_anom),np.nan, baseline_anomaly)
    
    baseline_ecmwf_file_list = sorted(glob(f'Data/ECMWF/soilw_bgrnd_processed/{region_name}/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
    baseline_ecmwf = xr.open_mfdataset(baseline_ecmwf_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()

    
    #Need to open a template of ECMWF to mask the np.nan values that
    return(obs_anomaly_SubX_format, baseline_anomaly, baseline_ecmwf, var_OUT, template_testing_only)


def open_only_testing_anomaly_baseline_files(region_name, week_lead, day_num, mask_anom, test_start, test_end):

    print(f'Loading template data from {test_start} to {test_end}')
    mask,mask_anom = load_mask(region_name)
    
    obs_anomaly_SubX_format_testing =xr.open_mfdataset(return_short_array(f'Data/GLEAM/RZSM_anomaly_reformat_SubX_format/{region_name}/RZSM_anomaly*',test_start,test_end)).sel(L=[day_num]).load()
    template_testing_only = obs_anomaly_SubX_format_testing.copy(deep=True)

    assert pd.to_datetime(start_).year == pd.to_datetime(end_).year, 'For this code to work properly, you must have the beginning and ending of the case study in the same year!'

    print(f'Loading observation, gefsv12 raw reforecast, and ecmwf raw reforecast during case study dates of {start_} to {end_}')
    obs_anomaly_SubX_format =xr.open_mfdataset(f'Data/GLEAM/RZSM_anomaly_reformat_SubX_format/{region_name}/RZSM_anomaly*{pd.to_datetime(start_).year}*.nc4').sel(L=[day_num]).sel(S=slice(start_,end_)).load()

    var_OUT = np.empty(shape=(obs_anomaly_SubX_format.Y.shape[0], obs_anomaly_SubX_format.X.shape[0])) #48x96
    
    #Mask the final output to be np.nan for ocean values
    var_OUT = np.where(mask_anom==1, np.nan, var_OUT)
    var_OUT[:,:] = 0
    
    #######################################   Reforecast baseline files   ###########################################################################
    # baseline_anomaly_file_list = sorted(glob('Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/RZSM*.nc'))
    if region_name =='CONUS':
        baseline_anomaly_file_list = sorted(glob(f'Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
    
    else:
        baseline_anomaly_file_list = sorted(glob(f'Data_{region_name}/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
        baseline_anomaly = xr.where(np.isnan(mask_anom),np.nan, baseline_anomaly)
    
    baseline_ecmwf_file_list = sorted(glob(f'Data/ECMWF/soilw_bgrnd_processed/{region_name}/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
    baseline_ecmwf = xr.open_mfdataset(baseline_ecmwf_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()

    
    #Need to open a template of ECMWF to mask the np.nan values that
    return(obs_anomaly_SubX_format, baseline_anomaly, baseline_ecmwf, var_OUT, template_testing_only)