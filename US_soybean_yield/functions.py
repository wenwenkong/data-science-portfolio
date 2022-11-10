import seaborn as sns
from matplotlib import pyplot as plt
import xarray as xr
import pandas as pd
from functools import reduce

def fetch_NLDAS(path, yr_Start, yr_End, month_number, month_suffix):
    """
    Combine all NLDAS data to one dataframe.
    
    Parameters
    ----------
    path: string
        path where the raw netcdf NLDAS datasets live
        in my case, the path is '/Users/wenwen/Downloads/CropYield/DATA/NLDAS/'
        
    yr_Start: int
        the beginning year of interest 
    
    yr_End: int
        the ending year of interest 
    
    month_number: list of strings 
        used to read in the months of interest from raw NLDAS .nc4 files
        for example: month_number = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        could also read in fewer months if that's desired 
        
    month_suffix: list of strings
        suffix to be added for each month's variable for notations 
        for example: 
        month_suffix = ['_Jan', '_Feb', '_Mar', '_Apr', '_May', '_Jun', '_Jul', '_Aug', '_Sep', '_Oct', '_Nov', '_Dec']
    
    Returns
    ----------
    Dataframe 
    
    """

    # Empty list for containing the dataframe generated for each year
    NLDAS_all_years         = []

    # Year list for the loop
    years                   = list(range(yr_Start, yr_End+1))

    # ***********************
    # Loop through years 
    # ***********************
    for year in years:
        #print('---- Processing NLDAS year '+str(year)+' ----')
        df_each_year     = []
    
        # *********************************************************
        # Loop through months
        # *********************************************************
        for  month, suffix in zip(month_number, month_suffix):
            file      = 'NLDAS_NOAH0125_M.A'+str(year)+month+'.020.nc.SUB.nc4'
            file_ds   = xr.open_dataset(path + file)
            file_df   = file_ds.to_dataframe()
            file_df   = file_df.reset_index()
    
            # Read in bnds 0 only, and sort values by the ascending order of lat and lon
            file_df_sorted = file_df[file_df['bnds'] == 0].sort_values(by = ['lat', 'lon'])
    
            # Drop the bnds and time bnds column
            # Rename the time column to year
            # Add suffix to the rest columns to denote the month
    
            file_df_sorted.drop(['bnds', 'time_bnds'], axis = 1, inplace = True)

            # *********************************************
            # Rename the column names - add monthly suffix
            # *********************************************
            column_names = list(file_df_sorted)
            for index in range(len(column_names)):
                if index <=1:
                    column_names[index] = column_names[index]
                elif index ==2:
                    column_names[index] = 'year'
                else:
                    column_names[index] += suffix
            
            # file_df_sorted - the dataframe for this year, this month
            file_df_sorted.columns = column_names
    
            # Change the value of 'year' column to year
            file_df_sorted['year'] = str(year)
        
            # Append the dataframe from this year, this month to df_each_year 
            df_each_year.append(file_df_sorted)
    
        
        # *************************************
        # Put all months to one dataframe
        # Note: don't do how='outer'
        # Using how = 'outer' will make the final_df 12 times larger (because here we are merging 12 months dataframes)
        # *************************************
        final_df = reduce(lambda  left,right: pd.merge(left,right, on=['lat', 'lon', 'year']), df_each_year)
    
    
        # Append this year's dataframe to the df_all_years list
        NLDAS_all_years.append(final_df)
    
        # Print out the progress
        print('NLDAS year '+str(year)+' is processed...')

    NLDAS_df = pd.concat(NLDAS_all_years)
    return NLDAS_df

def plot_corr_yield_current_climate(df, var, climate_dict, title):
    """
    Visualize the orrelation heatmaps between yield and climate variables.
    
    Parameters
    ----------
    df: dataframe
        dataframe containing the yield data and the climate variables
        
    var: list of strings
        names of climate variables for calculating the correlation with yield 
    
    climate_dict: dictionary
        user-defined dictionary including paird variable name and their corresponding standard name and units
        used to provide more informative subplot title 

    title: string
        user-defined suptitle for the panel plot
    
    Returns
    ----------
    panel correlation heatmaps 
    
    """
    
    plt.figure(figsize=(60,60))
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.1)
    plt.suptitle(title, fontsize=100, y = 0.95)

    sns.set(font_scale=3)

    ncols     = 2 # ---- set number of columns 
    nrows = len(var) // ncols + (len(var) % ncols > 0) # ---- calculate number of rows
    
    # loop through the length of var list and keep track of index
    for n, item in enumerate(var):
        # ---- add a new subplot iteratively using nrows and cols
        ax   = plt.subplot(nrows, ncols, n + 1)
    
        cols = [col for col in df.columns if col.startswith('yield') or col.startswith(item+'_')]
    
        res = sns.heatmap(df[cols].corr(), annot=True, fmt='.1f', cmap='RdBu_r', vmin=-1, vmax=1, ax=ax)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 30, rotation=90)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 30, rotation=0)
        res.xaxis.tick_top() # x axis on top
        res.xaxis.set_label_position('top')
        res.tick_params(length=0)
        ax.set_title(item + ' : '+climate_dict[item], fontsize=50, fontweight='bold', y =  1.1)

def plot_corr_climate(df, *args):    
    """
    Visualize the orrelation heatmaps between climate variables.
    
    Parameters
    ----------
    df: dataframe
        dataframe containing the climate variables
        
    args: list of paired tuples 
        paired climate variables for calculating the correlation with each other 
    
    Returns
    ----------
    panel correlation heatmaps 
    """
    
    plt.figure(figsize=(100,100), linewidth=40)
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.1)
    #plt.suptitle(title, fontsize=100, y = 1.0)

    sns.set(font_scale=5)

    ncols     = 2 # ---- set number of columns
    nrows = len(args) // ncols + (len(args) % ncols > 0) # ---- calculate number of rows
    
    for n, pair in enumerate(args):
        ax     = plt.subplot(nrows, ncols, n + 1)
        
        cols_1 = [col for col in df.columns if col.startswith(pair[0]+'_')]
        cols_2 = [col for col in df.columns if col.startswith(pair[1]+'_')]
        cols   = cols_1 + cols_2
    
        res = sns.heatmap(df[cols].corr(), annot=True, fmt='.1f', cmap='RdBu_r', vmin=-1, vmax=1, ax=ax)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 50, rotation=90)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 50, rotation=0)
        res.xaxis.tick_top() # x axis on top
        res.xaxis.set_label_position('top')
        res.tick_params(length=0)
        res.set_title(pair[0] + ' vs '+pair[1], fontsize=100, fontweight='bold', y =  1.1)

def plot_corr_climate_selected_months(df, *args):    
    """
    Visualize the orrelation heatmaps between climate variables.
    
    Parameters
    ----------
    df: dataframe
        dataframe containing the climate variables
        
    args: list 
        contains two lists and one string
        the two inner lists contain paired climate variables for calculating the correlation with each other 
    
    Returns
    ----------
    panel correlation heatmaps 
    """
    
    plt.figure(figsize=(100,100), linewidth=40)
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.1)
    #plt.suptitle(title, fontsize=100, y = 1.0)

    sns.set(font_scale=5)

    ncols     = 2 # ---- set number of columns
    nrows = len(args) // ncols + (len(args) % ncols > 0) # ---- calculate number of rows
    
    for n, pair in enumerate(args):
        ax     = plt.subplot(nrows, ncols, n + 1)
        
        cols   = pair[0] + pair[1]
    
        res = sns.heatmap(df[cols].corr(), annot=True, fmt='.1f', cmap='RdBu_r', vmin=-1, vmax=1, ax=ax)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 50, rotation=90)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 50, rotation=0)
        res.xaxis.tick_top() # x axis on top
        res.xaxis.set_label_position('top')
        res.tick_params(length=0)
        res.set_title(pair[2], fontsize=100, fontweight='bold', y =  1.1)
