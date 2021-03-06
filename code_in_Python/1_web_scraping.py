# -*- coding: utf-8 -*-
"""
Name: web_scaping.py
Purpose: Gather relevant data for predicting the cereal yield for each country
Tools: Selenium and Beautiful Soup for webscraping, Pandas and Numpy
References:      
    Selenium tutorial: https://selenium-python.readthedocs.io/navigating.html
    WebDriver for Chrome: http://chromedriver.chromium.org
"""

import numpy as np
import pandas as pd
import csv

import requests
import time

from bs4 import BeautifulSoup
from selenium import webdriver


def scrap_the_world_bank_ord_data(url):
    '''
    Download the csv file from The World Bank Organization
    
    Input argument: url in string
    '''
    chromedriver = "/Applications/Internet Software/chromedriver" 
    driver = webdriver.Chrome(chromedriver)
    driver.get(url)
    download_csv = driver.find_element_by_xpath('//*[@id="mainChart"]/aside/div/div[2]/div/p/a[1]')
    download_csv.click()
    driver.quit()


def scrap_temperature_rainfall_from_CCKP():
    '''
    Site: Climate Change Knowledge Portal
    Download excel files for monthly temperature and precipitation within
    years 1991-2015
    '''
    chromedriver = "/Applications/Internet Software/chromedriver" 
    driver = webdriver.Chrome(chromedriver)
    url = 'http://sdwebx.worldbank.org/climateportal/index.cfm?page=downscaled_data_download&menu=historical'
    driver.get(url)
    
    # pre=select time period 1991-2015
    variable = driver.find_element_by_id('historicalPeriodID')
    name = variable.find_element_by_class_name('histTimeSeries')
    options = name.find_elements_by_tag_name('option')
    options[-1].click()
    
    # set "variable" to either 1) temp or 2) precipitation (or rain fall)
    for v in range(0,2):     
        variable = driver.find_element_by_id('variableHistoricalID')
        name = variable.find_element_by_class_name('variablesHist')
        options = name.find_elements_by_tag_name('option')
        options[v].click() 
        
        # select country to download the excel file. 
        #    the filename is fixed, so macbook will append a _(index) BUT
        #    maximum index is 100 per folder. There are over 200 countries, 
        #    so you have mannually to divide the download into seperate sessions.
        #    For over 200 countries, and you need about 3 sessions.
        variable = driver.find_element_by_id('countryID')
        name = variable.find_element_by_class_name('country')
        options = name.find_elements_by_tag_name('option')
        for idx, option in enumerate(options): #Total 233 options
            if idx > 0:     #ignore first option, idx=0
                option.click()
                button = driver.find_elements_by_xpath('//*[@id="btnDiv"]/input')[0]
                button.click()
                # A popup windows ask users to choose the method of downloading
                #   the excel file
                download_file = driver.find_element_by_xpath('//*[@id="file1"]')
                download_file.click()
                time.sleep(5)
                # Click the cancel button to close the popup window
                cancel = driver.find_element_by_xpath('//*[@id="home"]/div[5]/div[11]/div/button/span')
                cancel.click()
                time.sleep(2)
    driver.quit()
    
    

def scrap_wheat_production_table_from_wiki():    
    '''
    Site: Wikipedia
    Scrap the data in the internationa
    '''   
    wheat_production_url = 'https://en.wikipedia.org/wiki/International_wheat_production_statistics'
    page = requests.get(wheat_production_url).text
    soup = BeautifulSoup(page, 'lxml')
    print(soup.prettify)
    
    wikitable = soup.find('table',{'class':'wikitable sortable'})
    country_links = wikitable.find_all('a')
    
    countries = []
    for link in country_links:
        name = link.get('title')
        if name is not None:
            countries.append(name)
    
    rows = np.array([])
    years = []
    row_links = wikitable.find_all('tr')
    for idx,link in enumerate(row_links):
        row = np.array([])
        if idx == 0: # the years
            cols = link.find_all('th')
            for ci, col in enumerate(cols):
                # ci==0 is ignore because the 'country' label name will be added
                #   when these data are converted to a data frame
                if ci > 0:
                    years.append(col.text[0:4])
        else:
            cols = link.find_all('td')
            for ci, col in enumerate(cols):
                if ci == 0:
                    if col.text.replace('\n','') == 'World total':
                        countries.append(col.text.replace('\n',''))
                else:
                    #row = np.append(row,str(col).replace('<td>','').replace('\n</td>',''))
                    if col.text.find('[') > -1:
                        row = np.append(row, col.text[0:col.text.find('[')])
                    else:
                        row = np.append(row, col.text.replace('\n',''))
        
        if len(rows) > 0:
            rows = np.vstack((rows, row))
        else: 
            rows = row
            
    # Use DataFrame to store the data and then save it to csv format
    df = pd.DataFrame()
    df['Country'] = countries
    df_wheat_prod_year = pd.concat([df,pd.DataFrame(rows, columns=list(years))], axis=1)
    df_wheat_prod_year.to_csv('wheat_production_year.csv', index=False)
    
    
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

    
if __name__ == "__main__":
    '''
    Site: The World Bank Organization
    
    Download data in csv format to your local 'Download' folder
    url1 = Employment in agriculture (% of total employment)
    url2 = Agricultural land (% of land area)
    url3 = Agricultural irrigated land (% of total agricultural land)
    url4 = Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)
    url5 = Agriculture, forestry, and fishing, value added (% of GDP)
    url6 = Cereal yield (kg per hectare)
    url7 = Fertilizer consumption (kilograms per hectare of arable land)
    url8 = GDP per capita (current US$)
    url9 = Infrastructure
    url10 = Land under cereal production (hectares)
    url11 = Agricultural machinery, tractors per 100 sq. km of arable land
    url12 = Pump price for diesel fuel (US$ per liter)
    url13 = Pump price for gasoline (US$ per liter)
    url14 = Rural population (% of total population)
    url15 = Population, total
    url16 = GDP (current US$)
    
    '''
    url1 = 'https://data.worldbank.org/indicator/SL.AGR.EMPL.ZS'
    url2 = 'https://data.worldbank.org/indicator/AG.LND.AGRI.ZS'
    url3 = 'https://data.worldbank.org/indicator/AG.LND.IRIG.AG.ZS'
    url4 = 'https://data.worldbank.org/indicator/er.h2o.fwag.zs'
    url5 = 'https://data.worldbank.org/indicator/NV.AGR.TOTL.ZS'
    url6 = 'https://data.worldbank.org/indicator/AG.YLD.CREL.KG'
    url7 = 'https://data.worldbank.org/indicator/AG.CON.FERT.ZS'
    url8 = 'https://data.worldbank.org/indicator/NY.GDP.PCAP.CD'
    url9 = 'https://data.worldbank.org/topic/infrastructure'
    url10 = 'https://data.worldbank.org/indicator/AG.LND.CREL.HA'
    url11 = 'https://data.worldbank.org/indicator/AG.LND.TRAC.ZS'
    url12 = 'https://data.worldbank.org/indicator/EP.PMP.DESL.CD'
    url13 = 'https://data.worldbank.org/indicator/EP.PMP.SGAS.CD'
    url14 = 'https://data.worldbank.org/indicator/SP.RUR.TOTL.ZS'
    url15 = 'https://data.worldbank.org/indicator/SP.POP.TOTL'
    url16 = 'https://data.worldbank.org/indicator/NY.GDP.MKTP.CD'
    
    for i in range(1,17):
        url = 'url'+str(i)
        scrap_the_world_bank_ord_data(url)
    
    '''
    Site: Climate Change Knowledge Portal
    Download excel files for monthly temperature and precipitation within
    years 1991-2015
    Then, write them into csv file(s)
    '''          
    scrap_temperature_rainfall_from_CCKP()
    write_min_max_avg_of_CCKP_rain_temp_xls_to_csv()
    
    
    '''
    Site: Wikipedia
    Scrap the data in the international wheat production statistics table
    and convert them into a dataframe
    '''
    scrap_wheat_production_table_from_wiki()
    

    