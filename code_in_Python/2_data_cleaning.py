#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: data_cleaning.py
Purpose: 
    1) extract relevant data from the csv files
    2) filter out the countries where any of the feature has no data for 
        25 years (1991=2015)
        
See See jupyter notebook (EDA.ipynb)

Tools: Pandas, Numpy, and other python modules
References:      
    https://stackoverflow.com/questions/52933862/sklearn-or-pandas-impute-missing-values-with-simple-linear-regression
    https://measuringu.com/handle-missing-data/
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict


# convert these features' csv files to DataFrames are in data/raw folder
csvfiles = ['Agricultural_land_percent.csv','Agricultural_value_percentGDP.csv',\
            'Total_population.csv','Cereal_yield_kg_per_hectare.csv',\
            'Employment_agriculture_percent.csv','Fertilizer_consumption_kg.csv',\
            'GDP_USD.csv', 'GDPpercapita_USD.csv','Machinery_per_100sqkm_arableland.csv',\
            'Pump_price_diesel_USDperliter.csv','Pump_price_gasoline_USDperliter.csv',\
            'Ruralpopulation_percent.csv','cereal_production_land_hectares.csv', \
            'tas_avg.csv','tas_min.csv','tas_max.csv','pr_avg.csv','pr_min.csv','pr_max.csv']

country_code = 'continent_country_list.csv'
df_countrycode = pd.read_csv(country_code, keep_default_na=False)
df_Aland = pd.read_csv(csvfiles[0])
df_Avalue = pd.read_csv(csvfiles[1])
df_Pop = pd.read_csv(csvfiles[2]) #
df_CkgH = pd.read_csv(csvfiles[3])
df_E = pd.read_csv(csvfiles[4])
df_F = pd.read_csv(csvfiles[5])
df_Gdp= pd.read_csv(csvfiles[6])
df_Gpc = pd.read_csv(csvfiles[7])
df_M = pd.read_csv(csvfiles[8])
df_Dpl = pd.read_csv(csvfiles[9])
df_Gpl = pd.read_csv(csvfiles[10])
df_R = pd.read_csv(csvfiles[11])
df_Cpl = pd.read_csv(csvfiles[12])
df_Tavg = pd.read_csv(csvfiles[13])
df_Tmin = pd.read_csv(csvfiles[14])
df_Tmax = pd.read_csv(csvfiles[15])
df_Ravg = pd.read_csv(csvfiles[16])
df_Rmin = pd.read_csv(csvfiles[17])
df_Rmax = pd.read_csv(csvfiles[18])


def extract_data_from_code(ccode, year_start, year_end, is_catagorical):
    '''
    This function combines the data of all the features that are stored
    in seperate csv files, for a given code, which can be either a country code
    or a continent/regional code
    
    Input arguments:
    ccode = a country code or a continent/regional code
    year_start = starting year of interest (i.e., 1991)
    year_end = ending year of interest (i.e., 2015)
    is_catagorical = if True, country code represents the country feature
            if False, country feature represented by latitude and longtitude
    '''
    df = pd.DataFrame()
    n = (year_end-year_start)+1     # number of years
    df['year'] = [str(y) for y in range(year_start,year_end+1)]
    if is_catagorical:
        df['code'] = [ccode]*n
    
    # Extract features of interest from the csv files 
    df['A'] = df_Aland[df_Aland['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['V'] = df_Avalue[df_Avalue['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['P'] = list(np.array(df_Pop[df_Pop['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0])/1000)
    df['Ckg'] = df_CkgH[df_CkgH['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['E'] = df_E[df_E['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['F'] = df_F[df_F['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['G'] = df_Gdp[df_Gdp['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['M'] = df_M[df_M['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Gc'] = df_Gpc[df_Gpc['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Dl'] = df_Dpl[df_Dpl['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Gl'] = df_Gpl[df_Gpl['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Cpl'] = df_Cpl[df_Cpl['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['R'] = df_R[df_R['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Tavg'] = df_Tavg[df_Tavg['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Tmin'] = df_Tmin[df_Tmin['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Tmax'] = df_Tmax[df_Tmax['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Ravg'] = df_Ravg[df_Ravg['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Rmin'] = df_Rmin[df_Rmin['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df['Rmax'] = df_Rmax[df_Rmax['Code']==ccode].loc[:,str(year_start):str(year_end)].values.tolist()[0]
    df.set_index('year', drop=True, inplace=True)
   
    return df



def check_any_column_nodata(ccode, year_start, year_end):
    '''
    This function checks if there are data in the columns (i.e., features) for
    a specific counry or regional code. 
    If there is no data in the entire column(s), it returns a boolean 
    
    Input arguments:
    ccode = a country code or a continent/regional code
    year_start = starting year of interest (i.e., 1991)
    year_end = ending year of interest (i.e., 2015)
    '''    
    n = (year_end-year_start)+1     # number of years
    
    c1n = df_CkgH[df_CkgH['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c2n = df_R[df_R['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c3n = df_Gpc[df_Gpc['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c4n = df_Gdp[df_Gdp['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c5n = df_E[df_E['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c6n = df_Pop[df_Pop['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c7n = df_M[df_M['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c8n = df_Dpl[df_Dpl['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c9n = df_Gpl[df_Gpl['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()
    c10n = df_Avalue[df_Avalue['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum() 
    c11n = df_R[df_R['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum() 
    
    if any((c/n) > 0.8 for c in [c1n, c2n, c3n, c4n, c5n, c6n, c7n, c8n, c9n, c10n, c11n]):
        return False
    else:
        return True
    
      
def check_country_has_CkgH(ccode, year_start, year_end):
    '''
    This function checks if there are data for target, cereal production. 
    
    Input arguments:
    ccode = a country code or a continent/regional code
    year_start = starting year of interest (i.e., 1991)
    year_end = ending year of interest (i.e., 2015)
    
    Return a boolean 
    '''        
    n = (year_end-year_start)+1     # number of years
    c = df_CkgH[df_CkgH['Code']==ccode].loc[:,str(year_start):str(year_end)].isna().values.sum()

    if (c/n) == 1:
        return False
    else:
        return True


def intersection_country_code():
    '''
    This function takes 3 sets of country codes and find the common/shared ones
    Return a list of common/shared country codes
    '''
    s1 = df_countrycode['Code'].values.tolist()
    s2 = df_Aland['Code'].values.tolist()
    s3 = df_Tavg['Code'].values.tolist()
    codes = list(set(s1) & set(s2) & set(s3))
    return codes  

def group_country_by_continent(codes):
    '''
    This function take the country codes and group them by the following 
    continents:
        AS = Asia
        EU = Europe
        OS = Oceania
        AF = Africa
        NA = North America
        SA = South America
    Return a dictionaty with continent code as the keys, and the values are 
        its conresponding country codes
    '''
    code_dict = defaultdict(list)
    for code in codes:
        cc = df_countrycode[df_countrycode['Code']==code]['Continent_Code'].values[0]
        code_dict[cc].append(code)
    return code_dict


if __name__ == "__main__":
   
    '''
    Extract features of interest per country for a selected time frame, and
    store them in a DataFrama. At the same time, find out which country
    is excluded due to no data
    Setting is_catagorical=True, making the countries as categorical variables
    '''
    df = pd.DataFrame()
    exclude_codes = []   # Record country coes with no data for specific column(s)
    codes_no_cereal = [] # Record country codes that have no data for the target
    codes = intersection_country_code()
    for ccode in codes:
        # Check any features (columns) has no or insufficient data
        #   exclude these countries if any features has no or insufficient data
        if check_any_column_nodata(ccode, 1991, 2015):
            df_temp = extract_data_from_code(ccode, 1991, 2015, is_catagorical=True)
            df = df.append([df_temp])
        else:
            exclude_codes.append(ccode)
        
        if not check_country_has_CkgH(ccode, 1991, 2015):
            codes_no_cereal.append(ccode)
            
    # Drop ['SLE', 'TLS', 'HTI', 'SWZ', 'MRT', 'LSO', 'CPV'] 
    #    because there isn't "F" value
    for c in ['SLE', 'TLS', 'HTI', 'SWZ', 'MRT', 'LSO', 'CPV']:
        df = df[df.code != c]
        
    # Final list of countries with data
    # After filter, only 118-7=111 countries left from total of 206 counties
    codes = list(set(codes) - set(exclude_codes)) 
    code_dict = group_country_by_continent(codes) 
    exclude_codes_dict = group_country_by_continent(exclude_codes)
    codes_no_cereal_dict = group_country_by_continent(codes_no_cereal)
    
    code_dict_list = {'code_dict':code_dict, 'codes_no_cereal_dict':codes_no_cereal_dict, 'exclude_codes_dict':exclude_codes_dict}
    
    '''
    # Pickle the DataFrama (Country are treated as categorical variables)
    # You can find them in data/interim folder
    with open('df_data_with_countrycode.pickle', 'wb') as write_to:
        pickle.dump(df, write_to)

    with open('code_dict_list.pickle', 'wb') as write_to:
        pickle.dump(code_dict_list, write_to)        
    '''
           
    # ******* Summary Statistics and Visualization ******
    # See See jupyter notebook (EDA.ipynb)
   
