
#for observations only (we still need to completely edit these functions but its a good start)


#Can change the difference weeks based on if we want to look at longer (or shorter) differenceing intervals


def return_std_and_mean_diff_across_years():

    std_daily_dict = {}
    mean_diff_dict = {}
    
    #First compute the rolling average based on the week differencing
    rolling_average = obs_anomaly.rolling(time=7*week_differencing, min_periods=7*week_differencing,center=False).mean()
    
    for idx,date in enumerate(time_index_short):
        # break
        #Grab all the same month and day values across all years
        
        #Need to add this because the leap year dates don't have enough values
        if date == pd.to_datetime('2000-02-29'):
            new_date = pd.to_datetime('2000-02-28')
        else:
            new_date = date
        
        #Select all the days across all years with the same month and day
        mask_current_week = (time_index_full.month == new_date.month) & (time_index_full.day == new_date.day)
        selected_data = obs_anomaly.isel(time=mask_current_week)

        #Now find the data from the previous week for each of those days
        if new_date == pd.to_datetime('2000-03-07'):
            leap_date = pd.to_datetime('2000-02-28')
        else:
            leap_date = new_date

        previous_week = leap_date - np.timedelta64(week_differencing,'W')
        mask_previous_week = (time_index_full.month == previous_week.month) & (time_index_full.day == previous_week.day)
        
        selected_data_previous = obs_anomaly.isel(time=mask_previous_week)

        #Sometimes we have a mis-match between years (specifically the number of data points, they must be equal!), so this fixes it
        if len(selected_data_previous.time.values) > len(selected_data.time.values):
            selected_data_previous = selected_data_previous.isel(time = slice(0,len(selected_data.time.values)))
        elif len(selected_data_previous.time.values) < len(selected_data.time.values):
            selected_data = selected_data.isel(time = slice(0,len(selected_data_previous.time.values)))


        #Now find the mean difference across all years and average
        mean_diff_across_years = np.nanmean(selected_data[putils.xarray_varname(selected_data)][:,:,:].values - selected_data_previous[putils.xarray_varname(selected_data)][:,:,:].values,axis=0)
        mean_diff_dict[f'{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}'] = mean_diff_across_years
        
        # rv1_data = selected_data.RZSM[:,:,:].values
        # rv2_data = selected_data_previous.RZSM[:,:,:].values
        # std_combined = xr.concat([selected_data, selected_data_previous], dim='RZSM_2').std(dim='time')
        
        # Calculate the covariance between rv1 and rv2 along the third dimension
        # covariance_rv1_rv2 = np.nancov(rv1_data.reshape(-1, rv1_data.shape[-1]), rv2_data.reshape(-1, rv2_data.shape[-1]))[0, 1]
        # Calculate the standard deviation of the difference between rv1 and rv2
        # std_dev_diff = np.sqrt(std_dev_rv1**2 + std_dev_rv2**2 - 2 * covariance_rv1_rv2)
        
        diff_across_years = selected_data[putils.xarray_varname(selected_data)][:,:,:].values - selected_data_previous[putils.xarray_varname(selected_data)][:,:,:].values
        
        std_ = np.nanstd(diff_across_years,axis=0)
                
        std_daily_dict[f'{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}'] = std_
        
    return(std_daily_dict,mean_diff_dict)


std_daily_dict,mean_diff_dict =return_std_and_mean_diff_across_years()




def get_current_and_previous_values_difference(date,file,week_differencing):
    #test 
    # file = obs_anomaly

    selected_data = obs_anomaly.sel(time=date)
    
    #Now find the data from the previous week for each of those days
    if (pd.to_datetime(date).month == 3) and (pd.to_datetime(date).day == 7):
        leap_date = pd.to_datetime(f'{pd.to_datetime(date).year}-02-28')
    else:
        leap_date = date

    previous_week = leap_date - np.timedelta64(week_differencing,'W')
    selected_data_previous = obs_anomaly.sel(time=previous_week)
    
    return(selected_data - selected_data_previous)

def rci_function_OBS_only(obs_anomaly,week_differencing):
    #Now we need to calculate the weekly difference betweeen weeks, then substract the mean, and divide by standard deviation
    rci = xr.zeros_like(obs_anomaly)

    for idx,date in enumerate(obs_anomaly.time.values):
        # break
        
        #We must begin only at MARCH
        if (pd.to_datetime(date).month == 3) and (pd.to_datetime(date).day <=7):
            # print(f'Working on date {date}')
            start_of_year = True
            #We don't have any weeks to work with before the 7th of MARCH
            rci[putils.xarray_varname(rci)][idx,:,:] = 0
        elif (pd.to_datetime(date).month in [3, 4, 5, 6, 7, 8, 9, 10, 11]) and (pd.to_datetime(date).year >=2000):
            # print(f'Working on date {date}')
            start_of_year = False
            # break

            diff_weeks = get_current_and_previous_values_difference(date,obs_anomaly,week_differencing)
            #Now standardize
            rci_standardized = (diff_weeks - mean_diff_dict[f'2000-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}']) / std_daily_dict[f'2000-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}']
            # plt.hist(rci_standardized.RZSM.values.flatten(),bins=30)
            # plt.show()
            
            #Now update RCI value
            subtract_ = xr.where(rci_standardized < -0.75,1,0)
            add_ = xr.where(rci_standardized > 0.75,1,0)
            
            #If the signs switch between postive and negative with RCI, then we reset rci to 0
            switch1 = np.where((rci_standardized[putils.xarray_varname(rci_standardized)] > 0) & (rci[putils.xarray_varname(rci)][idx - 7,:,:] < 0),2,0)
            switch2 = np.where((rci_standardized[putils.xarray_varname(rci_standardized)] < 0) & (rci[putils.xarray_varname(rci)][idx - 7,:,:] > 0),2,0)
            
            sub = np.where(subtract_[putils.xarray_varname(subtract_)] == 1, rci[putils.xarray_varname(rci)][idx-7,:,:] - np.sqrt(np.abs(rci_standardized[putils.xarray_varname(rci_standardized)])-0.75),0)
            add = np.where(add_[putils.xarray_varname(add_)] == 1, rci[putils.xarray_varname(rci)][idx-7,:,:] + np.sqrt(rci_standardized[putils.xarray_varname(rci_standardized)]-0.75),0)
            
            final = sub + add
            
            #Now switch back the data if the signs are oppositve
            final = np.where(switch1 != 2, final,0)
            final = np.where(switch2 != 2, final,0)
            
            rci[putils.xarray_varname(rci)][idx,:,:] = final
            
            # plt.hist(rci.RZSM[idx,:,:].values.flatten(),bins=30)
            # plt.show()
            
    return(rci)


#RCI save 
save_rci = f'{source}/GLEAM/RCI_index_{week_differencing}_week.nc'

if os.path.exists(save_rci):
    rci_output = xr.open_dataset(save_rci).load()
else:
    rci_output = rci_function_OBS_only(obs_anomaly,week_differencing)
    rci_output.to_netcdf(save_rci)


#Plot the distribution for the RCI file
rci_output










def convert_OBS_to_SubX_format(_date):  
# for _date in init_date_list:
    # var='RZSM_weighted'
    # _date=init_date_list[0]
    
    '''We are going to create new leads that are different than reforecast. The reasoning for this is that we want the actual weekly lags (and 1 day lag) and this will
    assist with future predictions within the deep learning model'''
    
    save_dir = f'{source}/GLEAM/RCI_reformat_SubX_format'
    os.system(f'mkdir -p {save_dir}')
    
    # for var in ['geopotential']:
    ref_dir = f'{source}/GEFSv12_reforecast/soilw_bgrnd' #Just use a single reference directory to serve as the template for file creation
  
    #Grab a single SubX to use as the template. Doesn't matter if it is the same variable or not or the same date
    fcst_file = glob(f'{ref_dir}/*soil*2000-01-05*')[0]
    op = xr.open_dataset(fcst_file)
    
    if region_name == 'CONUS':
        new_X_coords = [i+360 if i < 0 else i for i in op.X.values]
        op = op.assign_coords({'X':new_X_coords})
    
    open_date_SubX = putils.restrict_to_bounding_box(op,mask)
    out_file = xr.zeros_like(open_date_SubX)
    
    '''We are going to create a new lead day that represents the previous day before the forecast was initialized
    #New shape will be (1x11x48x48x96)
    This will include the day lag 1, and weekly lags 1-12'''
    
    file_shape = out_file[list(out_file.keys())[0]].shape

    save_date = f'{_date.year}-{_date.month:02}-{_date.day:02}'
    
    obs_file_name = f'RCI_{week_differencing}week_reformat_{save_date}.nc4'
    save_file = f'{save_dir}/{obs_file_name}'
    
    # if os.path.exists(save_file):
    if os.path.exists('this.out'):
        pass
    else:
        # os.system(f'rm {save_file}') #Just to avoid getting random duplicates
        print(f'Working on initialized day {_date} to find values integrating with SubX models, leads, & coordinates and saving data into {save_dir}.')
        
        for idx,i_lead in enumerate(out_file.L.values):
            # break

            date_val = pd.to_datetime(pd.to_datetime(_date) + dt.timedelta(days=int(i_lead)+0)) #Adding +1 may be suitable for other forecasts which predict the next day. But GEFSv12 predicts lead 0 as 12 UTC on the same date it is initialized
            #But be careful if you adapt this code to a new script. We are looking backwards in time from the first date.
                
            date_val = f'{date_val.year}-{date_val.month:02}-{date_val.day:02}'

            out_file[putils.xarray_varname(out_file)][0,:, idx, :, :] = \
                rci_output[putils.xarray_varname(rci_output)].sel(time = date_val).values

        var_OUT = xr.Dataset(
            data_vars = dict(
                rci = (['S','M','L','Y','X'],    out_file[list(out_file.keys())[0]].values),
            ),
            coords = dict(
                S = np.atleast_1d(_date),
                X = open_date_SubX.X.values,
                Y = open_date_SubX.Y.values,
                L = out_file.L.values,
                M = open_date_SubX.M.values,

            ),
            attrs = dict(
                Description = f'RCI values on the exact same date and grid \
                cell as EMC reforecast data. '),
        )                    

        var_OUT = var_OUT.astype(np.float32)
        
        var_OUT.to_netcdf(save_file)

    return(0)





####### RUN FUNCTION #######
for date in init_date_list:
    convert_OBS_to_SubX_format(date)



####### re open the rci files to analyze by lead


#open the RCI file in subx format
print('Loading the Observation RCI index file already computed.')
obs_rci = xr.open_mfdataset(f'{source}/GLEAM/RCI_reformat_SubX_format/RCI*{week_differencing}week*').sel(L=leads_).load()



def rci_reforecast_MEM(std_daily_dict, mean_diff_dict,week_differencing, reforecast_anomaly_MEM, obs_file, save_dir):

    os.system(f'mkdir -p {save_dir}')
    
    #testing
    # reforecast_anomaly_MEM = unet_anomaly
                        # Lead   0   ,    1,    ,   2      ,      3    ,     4     ,      5
    # output shape array([       nan, 0.0111007 , 0.01243984, 0.00984243, 0.01684886, 0.0191455 ])
    
    # obs_file = obs_rci.mean(dim='M')
    # save_dir = f'predictions/UNET/RCI/{experiment_name}'

    #Now go through each init day 

    #Take the mean
    
    
    # reforecast_anomaly_MEM = reforecast_anomaly_MEM.mean(dim='M').rolling(L=2,min_periods=2,center=False).mean() #this is for the 1-week RCI calculation
    # reforecast_anomaly_MEM.RZSM[0,:,10,10].values
    
    for idx,date in enumerate(reforecast_anomaly_MEM.S.values):
        # break
                
        datetime_dt = pd.to_datetime(date)
        save_date = f'{datetime_dt.year}-{datetime_dt.month:02}-{datetime_dt.day:02}'

        date_within_mean_std_dicts = f'2000-{datetime_dt.month:02}-{datetime_dt.day:02}'

        save_name_out = f'{save_dir}/RCI_MEM_{week_differencing}week_{save_date}.nc'
        #We need the very first lead as our beginning point with the RCI values from the observations
        rci_obs_to_save_over = obs_file.sel(S=date).copy(deep=True)
        rci_obs_to_save_over.rci[1:,:,:] = 0
        
        #Index only starts in March of every year and we must have it after the 7th bevcause that's when accumulation begins
        if (pd.to_datetime(date).month == 3) and (pd.to_datetime(date).day <=7):
            rci_obs_to_save_over.rci[0,:,:] = 0
            rci_obs_to_save_over = rci_obs_to_save_over.expand_dims({'S':1})
            rci_obs_to_save_over.to_netcdf(f'{save_name_out}')
            
        elif (datetime_dt.month in [3, 4, 5, 6, 7, 8, 9, 10, 11]):
            # break
            
            for idx2,lead in enumerate([6,13,20,27,34]):
                # break
                len_week_diff = np.arange(week_differencing)
                
                if idx2 in len_week_diff:
                    #We want to grab N weeks of the observations before the init date
                    obs_date_select = date - np.timedelta64(lead+1,'D')
                    week_diff = (reforecast_anomaly_MEM.sel(S=date, L=lead) - obs_anomaly.rename({'SMsurf':'RZSM'}).drop('season').sel(time=obs_date_select))
                    
                else:
                    #Now find the difference between the 2 weeks. The difference between the current week's forecast and the observation
                                   #Current day                                #Previous week
                    week_diff = reforecast_anomaly_MEM.sel(S=date, L=lead) - rci_obs_to_save_over.isel(L=idx2-week_differencing).rename({'rci':'RZSM'})
                    
                    #Now compute RCI
                    rci_standardized = (week_diff - mean_diff_dict[date_within_mean_std_dicts])/ std_daily_dict[date_within_mean_std_dicts]
                    rci_standardized.min()
                    rci_standardized.max()
                    rci_standardized.mean()
                    
                    #Now update RCI value
                    subtract_ = xr.where(rci_standardized < -0.75,1,0)
                    add_ = xr.where(rci_standardized > 0.75,1,0)
                    
                    #If the signs switch between postive and negative with RCI, then we reset rci to 0
                    switch1 = np.where((rci_standardized.RZSM > 0) & (rci_obs_to_save_over.isel(L=idx2-1).rci.values < 0),2,0)
                    switch2 = np.where((rci_standardized.RZSM < 0) & (rci_obs_to_save_over.isel(L=idx2-1).rci.values  > 0),2,0)
                    
                    sub = np.where(subtract_.RZSM.values == 1, rci_obs_to_save_over.isel(L=idx2-1).rci.values - (np.sqrt(np.abs(rci_standardized.RZSM.values) - 0.75)),0)
                    add = np.where(add_.RZSM.values == 1,  rci_obs_to_save_over.isel(L=idx2-1).rci.values + (np.sqrt(rci_standardized.RZSM.values-0.75)),0)

                    final = sub + add
            
                    #Now switch back the data if the signs are oppositve
                    final = np.where(switch1 != 2, final,0)
                    final = np.where(switch2 != 2, final,0)
                    
                    rci_obs_to_save_over.rci[idx2,:,:] = final
                    
            rci_obs_to_save_over = rci_obs_to_save_over.expand_dims({'S':1})
            rci_obs_to_save_over.to_netcdf(f'{save_name_out}')
                
        else:
            #don't need to update any dates that aren't between March through November
            rci_obs_to_save_over.rci[0,:,:] = 0
            rci_obs_to_save_over = rci_obs_to_save_over.expand_dims({'S':1})
            rci_obs_to_save_over.to_netcdf(f'{save_name_out}')

    return('Completed')


# Now calculate the RCI value based on previous observation differencing of standard deviation and day of year

def rci_reforecast(std_daily_dict, mean_diff_dict,week_differencing, reforecast_anomaly, obs_file, save_dir):

    os.system(f'mkdir -p {save_dir}')
    #testing
    # reforecast_anomaly = unet_anomaly
                        # Lead   0   ,    1,    ,   2      ,      3    ,     4     ,      5
    # output shape array([       nan, 0.0111007 , 0.01243984, 0.00984243, 0.01684886, 0.0191455 ])
    
    # obs_file = obs_rci
    # save_dir = f'predictions/UNET/RCI/{experiment_name}'

    #Now go through each init day 

    # reforecast_anomaly = reforecast_anomaly.rolling(L=2,min_periods=2,center=False).mean() #this is for the 1-week RCI calculation
    # reforecast_anomaly_MEM.RZSM[0,:,10,10].values
    
    for idx,date in enumerate(reforecast_anomaly.S.values):
        # break
                
        datetime_dt = pd.to_datetime(date)
        save_date = f'{datetime_dt.year}-{datetime_dt.month:02}-{datetime_dt.day:02}'

        date_within_mean_std_dicts = f'2000-{datetime_dt.month:02}-{datetime_dt.day:02}'

        save_name_out = f'{save_dir}/RCI_{week_differencing}week_{save_date}.nc'
        #We need the very first lead as our beginning point with the RCI values from the observations
        rci_obs_to_save_over = obs_file.sel(S=date).copy(deep=True)
        
        rci_obs_to_save_over.rci[:,:,:,:] = 0
        
        #Index only starts in March of every year and we must have it after the 7th bevcause that's when accumulation begins
        if (pd.to_datetime(date).month == 3) and (pd.to_datetime(date).day <=7):
            rci_obs_to_save_over.rci[:,0,:,:] = 0
            rci_obs_to_save_over = rci_obs_to_save_over.expand_dims({'S':1})
            rci_obs_to_save_over.to_netcdf(f'{save_name_out}')
            
        elif (datetime_dt.month in [3, 4, 5, 6, 7, 8, 9, 10, 11]):
            # break
            
            for idx2,lead in enumerate([6,13,20,27,34]):
                # break
                len_week_diff = np.arange(week_differencing)
                
                if idx2 in len_week_diff:
                    #We want to grab N weeks of the observations before the init date
                    obs_date_select = date - np.timedelta64(lead+1,'D')
                    week_diff = (reforecast_anomaly.sel(S=date, L=lead) - obs_anomaly.rename({'SMsurf':'RZSM'}).drop('season').sel(time=obs_date_select))
                    
                else:
                    #Now find the difference between the 2 weeks. The difference between the current week's forecast and the observation
                                   #Current day                                #Previous week
                    week_diff = reforecast_anomaly.sel(S=date, L=lead) - rci_obs_to_save_over.rci.isel(L=idx2-week_differencing)
                    
                #Now compute RCI
                rci_standardized = (week_diff - mean_diff_dict[date_within_mean_std_dicts])/ std_daily_dict[date_within_mean_std_dicts]
                rci_standardized.min()
                rci_standardized.max()
                rci_standardized.mean()
                
                try:
                    rci_standardized = rci_standardized.drop(['L','S'])
                except ValueError:
                    pass
                    
                #Now update RCI value
                subtract_ = xr.where(rci_standardized < -0.75,1,0)
                add_ = xr.where(rci_standardized > 0.75,1,0)
                
                #If the signs switch between postive and negative with RCI, then we reset rci to 0
                switch1 = np.where((rci_standardized[putils.xarray_varname(rci_standardized)] > 0) & (rci_obs_to_save_over.isel(L=idx2-1).rci.values < 0),2,0)
                switch2 = np.where((rci_standardized[putils.xarray_varname(rci_standardized)]< 0) & (rci_obs_to_save_over.isel(L=idx2-1).rci.values  > 0),2,0)
                
                sub = np.where(subtract_.RZSM.values == 1, rci_obs_to_save_over.isel(L=idx2-1).rci.values - (np.sqrt(np.abs(rci_standardized.RZSM.values) - 0.75)),0)
                add = np.where(add_.RZSM.values == 1,  rci_obs_to_save_over.isel(L=idx2-1).rci.values + (np.sqrt(rci_standardized.RZSM.values-0.75)),0)

                final = sub + add
        
                #Now switch back the data if the signs are oppositve
                final = np.where(switch1 != 2, final,0)
                final = np.where(switch2 != 2, final,0)
                
                rci_obs_to_save_over.rci[:,idx2,:,:] = final
                
            rci_obs_to_save_over = rci_obs_to_save_over.expand_dims({'S':1})
            rci_obs_to_save_over.to_netcdf(f'{save_name_out}')
                    
        else:
            #don't need to update any dates that aren't between March through November
            rci_obs_to_save_over.rci[:,0,:,:] = 0
            rci_obs_to_save_over = rci_obs_to_save_over.expand_dims({'S':1})
            rci_obs_to_save_over.to_netcdf(f'{save_name_out}')

    return('Completed')


#### map plots



   
# cmap = 'coolwarm'
def plot_case_study_rci(obs, unet, baseline, ecmwf, init_date):
    cmap = plt.get_cmap('bwr')    
    
    save_dir = f'Outputs/Case_studies/Southeast_US/RCI'
    os.system(f'mkdir -p {save_dir}')
        
    fig, axs = plt.subplots(
        nrows = 3, ncols= 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 10))
    axs = axs.flatten()
    
    init_date = pd.to_datetime(init_date)
    date = f'{init_date.year}-{init_date.month:02}-{init_date.day:02}'
    
    min_,max_ = get_min_max_of_files(obs, unet, baseline, ecmwf,date)
    # test_file = mae_rzsm_keys
    # for Subx original data
    v = np.linspace(-3, 3, 20, endpoint=True)

    pos = [i for i in v if i > 0]
    neg = [i for i in v if i < 0]

    neg.append(0)
    v = neg + pos
    
    lon = obs.X.values
    lat = obs.Y.values
    
    axs_start = 0
    for index_, lead in enumerate([20,27,34]):
        for data_to_plot,name in zip([obs, unet, baseline,ecmwf], ['GLEAM','UNET','GEFSv12','ECMWF']):
            # break
            data = return_array(file=data_to_plot,lead=index_, date=date)
        
            map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                          llcrnrlon=-128, urcrnrlon=-60, resolution='l')
            x, y = map(*np.meshgrid(lon, lat))
            # Adjust the text coordinates based on the actual data coordinates
        
            norm = TwoSlopeNorm(vmin=v[0], vcenter=0, vmax=v[-1])
        
            im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                                  transform=ccrs.PlateCarree(), cmap=cmap,norm=norm)
    
    
            # axs[idx].title.set_text(f'SubX Lead {lead*7}')
            gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                       linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
            gl.xlabels_top = False
            gl.ylabels_right = False
            if lead != 1:
                gl.ylabels_left = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            axs[axs_start].coastlines()
            # plt.colorbar(im)
            # axs[idx].set_aspect('auto', adjustable=None)
            axs[axs_start].set_aspect('equal')  # this makes the plots better
            axs[axs_start].set_title(f'{name} Lead {lead}',fontsize=15)
            axs_start+=1
    cbar_ax = fig.add_axes([0.05, -0.05, .9, .04])
    
    # Draw the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    fig.suptitle(f'Init date: {date}', fontsize=30)
    fig.tight_layout()
    
    plt.savefig(f'{save_dir}/Southeast_{week_differencing}week_RCI_init{date}.png',bbox_inches='tight')
    plt.show()



