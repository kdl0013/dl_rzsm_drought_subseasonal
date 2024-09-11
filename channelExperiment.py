#!/usr/bin/env python3

import xarray as xr
import os
from glob import glob
import pandas as pd
import numpy as np
import functions as f

def return_experiment_name(include_reforecast_or_not,num_lags_obs_RZSM,include_lags_obs_pwat_spfh_tmax,addtl_experiment,experiment_test):
    
    if experiment_test == 26:
        return('EX26')
    
    elif (include_reforecast_or_not == True) and (num_lags_obs_RZSM == 0) and (include_lags_obs_pwat_spfh_tmax == False):
        if experiment_test == 1:
            experiment_name = 'EX13'
        else:
            experiment_name = 'EX0'
        
    elif (include_reforecast_or_not == False) and (num_lags_obs_RZSM != 0) and (include_lags_obs_pwat_spfh_tmax == True):
        
        experiment_name = ''
        if num_lags_obs_RZSM == 3:
            if addtl_experiment == True and experiment_test == 1:
                experiment_name = 'EX14'
            else:
                experiment_name = 'EX1'
        elif num_lags_obs_RZSM == 6:
            if addtl_experiment == True and experiment_test == 2:
                experiment_name = 'EX15'
            else:
                experiment_name = 'EX2'      
        elif num_lags_obs_RZSM == 9:
            if addtl_experiment == True and experiment_test == 3:
                experiment_name = 'EX16'
            else:
                experiment_name = 'EX3'       
        elif num_lags_obs_RZSM == 12:
            if addtl_experiment == True and experiment_test == 4:
                experiment_name = 'EX17'
            else:
                experiment_name = 'EX4'
        else:
            assert len(experiment_name) != 0, 'Not a valid number for num_lags_obs_RZSM for our experiments. num_lags_obs_RZSM must be 3, 6, 9, 12'
            
            
    elif (include_reforecast_or_not == True) and (num_lags_obs_RZSM != 0) and (include_lags_obs_pwat_spfh_tmax == False):
        
        experiment_name = ''
        if num_lags_obs_RZSM == 3:
            if experiment_test == 1:
                experiment_name = 'EX18'
            elif experiment_test == 5:
                experiment_name = 'EX22'
            else:
                experiment_name = 'EX5'
        elif num_lags_obs_RZSM == 6:
            if experiment_test == 2:
                experiment_name = 'EX19'
            elif experiment_test ==6:
                experiment_name = 'EX23'
            else:
                experiment_name = 'EX6'        
        elif num_lags_obs_RZSM == 9:
            if experiment_test == 3:
                experiment_name = 'EX20'
            elif experiment_test ==7:
                experiment_name = 'EX24'
            else:
                experiment_name = 'EX7'        
        elif num_lags_obs_RZSM == 12:
            if experiment_test == 4:
                experiment_name = 'EX21'
            elif experiment_test == 8:
                experiment_name = 'EX25'
            else:
                experiment_name = 'EX8'
        else:
            assert len(experiment_name) != 0, 'Not a valid number for num_lags_obs_RZSM for our experiments. num_lags_obs_RZSM must be 3, 6, 9, 12'

    elif (include_reforecast_or_not == True) and (num_lags_obs_RZSM != 0) and (include_lags_obs_pwat_spfh_tmax == True):
        
        experiment_name = ''
        if num_lags_obs_RZSM == 3:
            if experiment_test == 0:
                experiment_name = 'EX9'
            elif experiment_test == 1:
                experiment_name = 'EX27'
            elif experiment_test == 2:
                experiment_name = 'EX29'
        elif num_lags_obs_RZSM == 6:
            if experiment_test == 0:
                experiment_name = 'EX10'
            elif experiment_test == 1:
                experiment_name = 'EX28' 
        elif num_lags_obs_RZSM == 9:
            experiment_name = 'EX11'        
        elif num_lags_obs_RZSM == 12:
            experiment_name = 'EX12'
        else:
            assert len(experiment_name) != 0, 'Not a valid number for num_lags_obs_RZSM for our experiments. num_lags_obs_RZSM must be 3, 6, 9, 12'

    return(experiment_name)


def return_num_day_lags_from_weekly_lags(num_lags_obs_RZSM):
    '''Assuming num_lags_obs_RZSM == 3, we need to get the data from specific day lags. 
    The first channel will be the day lag -1, while the other channels will be week lags 1-2 (which would be -14 and -21)'''
    if num_lags_obs_RZSM == 0:
        combined_data = None
    else:
        day_lag = -1
        week_lag = np.arange(1,num_lags_obs_RZSM) #start with weekly lags
        #Now combine the two different sets of numbers
        combined_data = []
        combined_data.append(day_lag)
        for i in week_lag:
            if i == -1:
                i=i
            else:
                i = -7*i
            combined_data.append(i)
    return(combined_data)


def return_channel_list(lead, experiment_name,lag_integer_list,observation_lag_list_not_RZSM, RZSM_or_Tmax_or_both):
    
    var_list = ['pwat','spfh','tmax','diff_temp','z200']
    
    '''Add only reforecast input data. This is a simple bias correction with no observations'''
        
    if experiment_name == 'EX0':
        channel_list = []
        if RZSM_or_Tmax_or_both == 'RZSM':
            for i in range(0,lead+1):
                channel_list.append(f'RZSM_ref_lead{i}')
                
        elif RZSM_or_Tmax_or_both == 'both':
            for i in range(0,lead+1):
                channel_list.append(f'RZSM_ref_lead{i}')
            for i in range(0,lead+1):
                channel_list.append(f'tmax_ref_lead{i}')

  

    elif (experiment_name == 'EX1') or (experiment_name == 'EX2') or (experiment_name == 'EX3') or (experiment_name == 'EX4'):
        channel_list = [f'RZSM_obs_lag{i}' for i in lag_integer_list]
        
        for var in var_list:
            for i in observation_lag_list_not_RZSM:
                channel_list.append(f'{var}_obs_lag{i}')
        
        if lead ==0:
            if RZSM_or_Tmax_or_both == 'RZSM':
                channel_list.append(f'RZSM_ref_lead{lead}')
            elif RZSM_or_Tmax_or_both == 'both':
                channel_list.append(f'RZSM_ref_lead{lead}')
                channel_list.append(f'tmax_ref_lead{lead}')
                
        if lead == 1:
            channel_list.append(f'RZSM_ref_lead1')
            channel_list.append(f'tmax_ref_lead1')
        else:
            for i in range(1,lead):
                channel_list.append(f'RZSM_prediction_lead{i}')
                channel_list.append(f'tmax_prediction_lead{i}')
        
    elif (experiment_name == 'EX5') or (experiment_name == 'EX6') or (experiment_name == 'EX7') or (experiment_name == 'EX8'):
        channel_list = [f'RZSM_obs_lag{i}' for i in lag_integer_list]
        
        for i in observation_lag_list_not_RZSM:
            channel_list.append(f'tmax_obs_lag{i}')
        
        for i in range(1,lead+1):
            channel_list.append(f'RZSM_ref_lead{i}')
            channel_list.append(f'tmax_ref_lead{i}')
        
            
    elif (experiment_name == 'EX9') or (experiment_name == 'EX10') or (experiment_name == 'EX11') or (experiment_name == 'EX12'):
        channel_list = [f'RZSM_obs_lag{i}' for i in lag_integer_list]

        for var in var_list:
            for i in observation_lag_list_not_RZSM:
                channel_list.append(f'{var}_obs_lag{i}')
        
        if lead ==1:
            channel_list.append('RZSM_ref_lead1')
            channel_list.append('tmax_ref_lead1')
        else:
            for i in range(1,lead):
                channel_list.append(f'RZSM_prediction_lead{i}')
                channel_list.append(f'tmax_prediction_lead{i}')
                
            channel_list.append(f'RZSM_ref_lead{lead}')
            channel_list.append(f'tmax_ref_lead{lead}')
            
    return(channel_list)