#!/usr/bin/env python3
import xarray as xr
import os

global dir
dir = '/glade/work/klesinger/FD_RZSM_deep_learning'

def load_CONUS_mask():
    #Load contiguous united states mask. This can be any arbitrary mask for files
    return(xr.open_dataset(f'{dir}/Data/CONUS_mask/CONUS_mask.nc'))

def load_Australia_mask():
    #This will be useful for making sure that we have the land regions with values and ocean is masked.
    #This is just 1 day from the GLEAM dataset
    return(xr.open_dataset(f'{dir}/Data_australia/GLEAM/australia_mask.nc4'))    
#Create a new mask for australia

def load_china_mask():
    #This will be useful for making sure that we have the land regions with values and ocean is masked.
    #This is just 1 day from the GLEAM dataset
    return(xr.open_dataset(f'{dir}/Data_china/GLEAM/china_mask.nc4')) 

#Create a new mask for Northern Hemisphere


def load_mask(region_name):
    #Load contiguous united states mask. This can be any arbitrary mask for files
    if region_name == 'CONUS':
        return(xr.open_dataset(f'{dir}/Data/CONUS_mask/CONUS_mask.nc'))
    elif region_name == 'australia':
        return(xr.open_dataset(f'{dir}/Data_australia/GLEAM/australia_mask.nc4'))
    elif region_name == 'china':
        return(xr.open_dataset(f'{dir}/Data_china/GLEAM/china_mask.nc4'))

def load_region_mask(region_name):
    #Load contiguous united states mask. This can be any arbitrary mask for files
    if region_name == 'CONUS':
        correct_shape = xr.open_dataset(f'{dir}/Data/CONUS_mask/CONUS_mask.nc')
        region_mask = xr.open_dataset(f'{dir}/Data/CONUS_mask/region_CONUS_mask.nc4')
        #Now select the correct dimensions
        return(region_mask.sel(latitude=correct_shape.Y.values,longitude=correct_shape.X.values)['NCAregions_mask'])
    elif region_name == 'australia':
        return(xr.open_dataset(f'{dir}/Data_australia/GLEAM/australia_mask.nc4'))