#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: data_cleaning.py
Purpose: 
    1) extract relevant data from multiple excel files or csv files
    2) filter out the countries where any of the feature has no data for 
        25 years (1991=2015)
    3) Apply imputation methods (subtraction and linear regression) to handle
        the missing records (i.e., NaN)
Tools: Pandas, Numpy, seaborn, matplotlib, and other python modules
References:      
    https://stackoverflow.com/questions/52933862/sklearn-or-pandas-impute-missing-values-with-simple-linear-regression
    https://measuringu.com/handle-missing-data/
"""

import pandas as pd
import numpy as np
import csv
import pickle
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt



def write_min_max_avg_of_CCKP_rain_temp_xls_to_csv():
    '''
    This function extracts the minimum, maximum, and average temperatures and 
    precipitations for each year and all the countries from the 200+ excel 
    files, and write them to the csv file(s).
    '''
    # Read a list of country names and continents from a csv file
    country_code = 'continent_country_list.csv'
    df_countrycode = pd.read_csv(country_code)

    # Gather all the folders that contain the monthly temperature and 
    # precipitation excelt tiles doanlowded from the CCKP website.
    tempfolder = ['dump1/','dump2/','dump3/']
    years = [str(y) for y in range(1991,2016)]
        
    with open('tas_min.csv', mode='w') as f1, open('tas_max.csv', mode='w') as f2, \
        open('tas_avg.csv', mode='w') as f3, open('pr_min.csv', mode='w') as g1, \
        open('pr_max.csv', mode='w') as g2, open('pr_avg.csv', mode='w') as g3:
        
        # write the column headers for each csv file
        tas_min_writer = csv.writer(f1, delimiter = ',')
        tas_min_writer.writerow(['Country', 'Code'] + years)
        tas_max_writer = csv.writer(f2, delimiter = ',')
        tas_max_writer.writerow(['Country', 'Code'] + years)
        tas_avg_writer = csv.writer(f3, delimiter = ',')
        tas_avg_writer.writerow(['Country', 'Code'] + years)
        
        pr_min_writer = csv.writer(g1, delimiter = ',')
        pr_min_writer.writerow(['Country', 'Code'] + years)
        pr_max_writer = csv.writer(g2, delimiter = ',')
        pr_max_writer.writerow(['Country', 'Code'] + years)
        pr_avg_writer = csv.writer(g3, delimiter = ',')
        pr_avg_writer.writerow(['Country', 'Code'] + years)
        
        for folder in tempfolder:
            if folder is 'dump3/':
                maxfile = 29
            else:
                maxfile = 101
                
            for fileind in range(0,maxfile):
                df_tas = pd.read_excel(folder+'tas_1991_2015 '+'('+str(fileind)+').xls')
                df_pr = pd.read_excel(folder+'pr_1991_2015 '+'('+str(fileind)+').xls')
                
                if not df_tas.empty or not df_pr.empty:
                    tmins = []
                    pmins = []
                    tmaxs = []
                    pmaxs = []
                    tavgs = []
                    pavgs = []
                    for year in years:
                        tsubdata = df_tas[df_tas['\tYear']==int(year)]
                        psubdata = df_pr[df_pr['\tYear']==int(year)]
                    
                        tcode = tsubdata.iloc[0][' Country']
                        #pcode = psubdata.iloc[0][' Country']
                        
                        if sum(df_countrycode['Code']==tcode) == 0 :
                            if tcode == 'ZAR':  # Special case for COD
                                code = 'COD' 
                                country = df_countrycode[df_countrycode['Code']==code].iloc[0]['Country']
                            elif tcode == 'XRK':    # Missing country code
                                country = 'Kosovo'
                            elif tcode == 'ROM':    # Missing country code
                                country = 'Romania'
                            else:             # Print any missing country code
                                print("Error:", tcode) 
                        else:
                            # Identify the country name from the country code
                            country = df_countrycode[df_countrycode['Code']==tcode].iloc[0]['Country']
                        
                        # Calculate the minimum, maximum, and average temperature
                        # of a specific year for a specific country
                        tmins.append(pd.np.min(tsubdata['tas'])) 
                        tmaxs.append(pd.np.max(tsubdata['tas'])) 
                        tavgs.append(pd.np.mean(tsubdata['tas'])) 
                        
                        # Calculate the minimum, maximum, and average precipitation
                        # of a specific year for a specific country
                        pmins.append(pd.np.min(psubdata['pr'])) 
                        pmaxs.append(pd.np.max(psubdata['pr'])) 
                        pavgs.append(pd.np.mean(psubdata['pr'])) 
                        
                    # Write the calculated temperature and precipitation
                    #   to theor coresponding csv files
                    tas_min_writer.writerow([country, tcode] + tmins)
                    tas_max_writer.writerow([country, tcode] + tmaxs)
                    tas_avg_writer.writerow([country, tcode] + tavgs)
                    
                    pr_min_writer.writerow([country, tcode] + pmins)
                    pr_max_writer.writerow([country, tcode] + pmaxs)
                    pr_avg_writer.writerow([country, tcode] + pavgs)
                
    # Close all csv objects (or files)            
    f1.close()
    f2.close()
    f3.close()
    
    g1.close()
    g2.close()
    g3.close()    

# convert these features' csv files to DataFrames
csvfiles = ['Agricultural_land_percent.csv','Agricultural_value_percentGDP.csv',\
            'Total_population.csv','Cereal_yield_kg_per_hectare.csv',\
            'Employment_agriculture_percent.csv','Fertilizer_consumption_kg.csv',\
            'GDP_USD.csv', 'GDPpercapita_USD.csv','Machinery_per_100sqkm_arableland.csv',\
            'Pump_price_diesel_USDperliter.csv','Pump_price_gasoline_USDperliter.csv',\
            'Ruralpopulation_percent.csv','cereal_production_land_hectares.csv', \
            'tas_avg.csv','tas_min.csv','tas_max.csv','pr_avg.csv','pr_min.csv','pr_max.csv']

country_code = 'continent_country_list.csv'
country_centroid = 'country_centroids.csv'
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
df_Cent = pd.read_csv(country_centroid)


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
    else:
        df['long'] = df_Cent[df_Cent['Code']==ccode]['Longitude'].values[0]
        df['lat'] = df_Cent[df_Cent['Code']==ccode]['Latitude'].values[0]
    
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
   
    # Apply imputation methods: Regression and Substitution
    df['Dl'].interpolate(method='linear', inplace=True, limit_direction="both")
    df['Gl'].interpolate(method='linear', inplace=True, limit_direction="both")
    df['M'].interpolate(method='linear', inplace=True, limit_direction="both")
    df['P'].interpolate(method='linear', inplace=True, limit_direction="both")
    df['R'].interpolate(method='linear', inplace=True, limit_direction="both")
    df['F'].fillna(0,inplace=True)
    df['V'].fillna(df['V'].min(),inplace=True)
    df['A'].fillna(df['A'].min(),inplace=True)
    df['Ckg'].fillna(df['Ckg'].min(),inplace=True)
    df['Cpl'].fillna(df['Cpl'].min(),inplace=True)
    df['G'].fillna(df['G'].min(),inplace=True)
    df['Gc'].fillna(df['Gc'].min(),inplace=True)
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
    elif df_Cent[df_Cent['Code']==ccode].empty:
        return False
    else:
        return True
    
      
def country_no_CkgH(ccode, year_start, year_end):
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
    elif df_Cent[df_Cent['Code']==ccode].empty:
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
        if check_any_column_nodata(ccode, 1991, 2015):
            df_temp = extract_data_from_code(ccode, 1991, 2015, is_catagorical=True)
            df = df.append([df_temp])
        else:
            exclude_codes.append(ccode)
        
        if not country_no_CkgH(ccode, 1991, 2015):
            codes_no_cereal.append(ccode)
            
    # Final list of countries with data
    # After filter, only 118 countries
    codes = list(set(codes) - set(exclude_codes)) 
    code_dict = group_country_by_continent(codes) 
    
    # ******* Summary Statistics and Visualization ******
    df.info()
    
    df.describe()
    
    # Plot the matrix of correlations with heatmap
    sns.heatmap(df.corr(), cmap="seismic", annot=False, vmin=-1, vmax=1)
    #plt.savefig('heatmap_EDA.png', dpi=300)
    #plt.clf()
    plt.show()
    
    # Create pair plots of the matrix of correlations
    sns.pairplot(df, height=1.2, aspect=1.5)
    plt.show()
    
    # Plot each feature's distribution
    features = ['A', 'V', 'P', 'E', 'F', 'M', 'Gc', 'G', 'Dl', 'Gl', 'Cpl', 'R', 'Tavg', 'Ravg']
    for feature in features:
        sns.distplot(df[feature],bins=10, norm_hist=True, color='g')
        plt.show()
    

    
    # Only retains features, average temperature and precipitation (Tavg and Ravg)
    df.drop(columns=['Tmin','Tmax','Rmin','Rmax' ], inplace=True)
    df.drop(columns=['G' ], inplace=True)   #drop GDP
    df.head(3)
    
    # Pickle the DataFrama (Country are treated as categorical variables)
    with open('country_14features.pickle', 'wb') as write_to:
        pickle.dump(df, write_to)


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
        if check_any_column_nodata(ccode, 1991, 2015):
            df_temp = extract_data_from_code(ccode, 1991, 2015, is_catagorical=False)
            df = df.append([df_temp])
        else:
            exclude_codes.append(ccode)
        
        if not country_no_CkgH(ccode, 1991, 2015):
            codes_no_cereal.append(ccode)
            
    # Final list of countries with data
    # After filter, only 118 countries
    codes = list(set(codes) - set(exclude_codes)) 
    code_dict = group_country_by_continent(codes)         
    # Pickle the DataFrama (countries are represented by latitute and longitude)
    with open('country_14features_latlong.pickle', 'wb') as write_to:
        pickle.dump(df, write_to)

    

    
    
