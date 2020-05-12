#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:53:59 2020

@author: fhall
"""

### imports
import datetime
import eemeter
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from statistics import mean
from eemeter_baseline import *

font = {'weight':'normal','size':20}
plt.rc('font', **font)
plt.rc('figure', figsize=(15, 10))
plt.rc('xtick.major', pad=8) # xticks too close to border!
plt.rc('ytick.major', pad=10) # xticks too close to border!
plt.rc('xtick',labelsize='small')
plt.rc('ytick',labelsize='small')


##############################################################################
#### LOAD & FORMAT DATA ######################################################
##############################################################################

# load 3 datasets
temperature = eemeter.temperature_data_from_csv("data/temperature.csv", freq = 'hourly')
t3_ami = pd.read_csv('data/t3_measure_ami.csv')
t3_daily = pd.read_csv('data/t3_measure_daily.csv')


# split ami out by measure type
aev_ami = t3_ami[t3_ami['Type'] == 'AEV']
phev_ami = t3_ami[t3_ami['Type'] == 'PHEV']
cchp_ami = t3_ami[t3_ami['Type'] == 'CCHP']
ebike_ami = t3_ami[t3_ami['Type'] == 'eBike']

aev_ami[['start', 'value']].to_csv('data/aev_ami.csv')
phev_ami[['start', 'value']].to_csv('data/phev_ami.csv')
cchp_ami[['start', 'value']].to_csv('data/cchp_ami.csv')
ebike_ami[['start', 'value']].to_csv('data/ebike_ami.csv')


# split daily usage out by measure type
aev_daily = t3_daily[t3_daily['Type'] == 'AEV']
phev_daily = t3_daily[t3_daily['Type'] == 'PHEV']
cchp_daily = t3_daily[t3_daily['Type'] == 'CCHP']
ebike_daily = t3_daily[t3_daily['Type'] == 'eBike']

aev_daily[['start', 'value']].to_csv('data/aev_daily.csv')
phev_daily[['start', 'value']].to_csv('data/phev_daily.csv')
cchp_daily[['start', 'value']].to_csv('data/cchp_daily.csv')
ebike_daily[['start', 'value']].to_csv('data/ebike_daily.csv')


# the dates of an analysis "blackout" period during which a project was performed.
install_start = datetime.datetime(2018, 4, 26, tzinfo=pytz.UTC)
install_end = datetime.datetime(2019, 4, 26, tzinfo=pytz.UTC)


#### CREATE BASELINES ########################################################

hourly_metrics = []
daily_metrics = []

# hourly ami baselines
aev_baseline, aev_metrics, aev_model_results = eemeter_baseline_ami('data/aev_ami.csv', 
                                                  temperature, 
                                                  install_start,
                                                  install_end)

phev_baseline, phev_metrics, phev_model_results = eemeter_baseline_ami('data/phev_ami.csv', 
                                                  temperature, 
                                                  install_start,
                                                  install_end)

cchp_baseline, cchp_metrics, cchp_model_results = eemeter_baseline_ami('data/cchp_ami.csv', 
                                                  temperature, 
                                                  install_start,
                                                  install_end)

ebike_baseline, ebike_metrics, ebike_model_results = eemeter_baseline_ami('data/ebike_ami.csv', 
                                                  temperature, 
                                                  install_start,
                                                  install_end)



# daily eemeter models
aev_daily_baseline, aev_daily_metrics, aev_daily_model_results = eemeter_baseline_daily('data/aev_daily.csv', temperature, install_start, install_end)
phev_daily_baseline, phev_daily_metrics, phev_daily_model_results = eemeter_baseline_daily('data/phev_daily.csv', temperature, install_start, install_end)
cchp_daily_baseline, cchp_daily_metrics, cchp_daily_model_results = eemeter_baseline_daily('data/cchp_daily.csv', temperature, install_start, install_end)
ebike_daily_baseline, ebike_daily_metrics, ebike_daily_model_results = eemeter_baseline_daily('data/ebike_daily.csv', temperature, install_start, install_end)



# combine load growth
load_growth = pd.concat([aev_baseline['metered_savings'],
                     phev_baseline['metered_savings'],
                     cchp_baseline['metered_savings'],
                     ebike_baseline['metered_savings'],
                     ebike_baseline['temp']], 
                    axis=1)
load_growth.columns = ['AEV','PHEV','CCHP','eBike','tempF']

# combine results
measures = ['AEV','PHEV','CCHP','eBIKE']
hourly_metrics = [aev_metrics, phev_metrics, cchp_metrics, ebike_metrics]
daily_metrics = [aev_daily_metrics, phev_daily_metrics, cchp_daily_metrics, ebike_daily_metrics]


# Model Metrics: Hourly
print('Hourly'.ljust(5),'adj_r2','rmse_adj', 'new_load',sep='\t')
for i in range(4):
    print(measures[i].ljust(5),"%.3f" % hourly_metrics[i][0], "%.3f" % hourly_metrics[i][1], "%.0f" %hourly_metrics[i][2], sep='\t')

print('\n') # spacing

# Model Metrics: Daily
print('Daily'.ljust(5),'adj_r2','rmse_adj', 'new_load',sep='\t')
for i in range(4):
    print(measures[i].ljust(5),"%.3f" % daily_metrics[i][0], "%.3f" % daily_metrics[i][1], "%.0f" %daily_metrics[i][2], sep='\t')


##############################################################################
#### VISUALIZATION ###########################################################
##############################################################################

temp_bins = load_growth.tempF//20
temp_labels = ['below 0','0 to 19','20 to 39','40 to 59','60 to 79', 'above 80']


# determine mean usage by temperature bin
load_growth_grouped_mean = load_growth.groupby(temp_bins).agg(['mean'])
load_growth_grouped_mean['temp_bin'] = temp_labels

# determine total usage by temperature bin
load_growth_grouped_total = load_growth.groupby(temp_bins).agg(['sum'])
load_growth_grouped_total['temp_bin'] = temp_labels

# determine max usage by temperature bin
load_growth_grouped_max = load_growth.groupby(temp_bins).agg(['max'])
load_growth_grouped_max['temp_bin'] = temp_labels


# PLOT 1
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
colors = ["windows blue", "amber", "greyish", "faded green"]

# sublot 1: mean
load_growth_grouped_mean.plot.bar(x='temp_bin',y=['AEV','CCHP','PHEV','eBike'], 
                       label = ['AEV','CCHP','PHEV','eBike'],
                       color = sns.xkcd_palette(colors),
                       ax=axes[0],
                       legend=False)
axes[0].tick_params(rotation=0)
axes[0].set_ylabel('mean kWh growth')
fig.legend(loc="center right")

# subplot 2: total
load_growth_grouped_total.plot.bar(x='temp_bin',y=['AEV','CCHP','PHEV','eBike'], 
                       label = ['AEV','CCHP','PHEV','eBike'],
                       color = sns.xkcd_palette(colors),
                       ax=axes[1],
                       legend=False)
axes[1].tick_params(rotation=0)
axes[1].set_ylabel('total kWh growth')

# subplot 3: max
load_growth_grouped_max.plot.bar(x='temp_bin',y=['AEV','CCHP','PHEV','eBike'], 
                       label = ['AEV','CCHP','PHEV','eBike'],
                       color = sns.xkcd_palette(colors),
                       ax=axes[2],
                       legend=False)
axes[2].tick_params(rotation=0)
axes[2].set_xlabel('temperature bins (F)')
axes[2].set_ylabel('max kWh growth')




### LOAD PROFILE
hour = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

aev_load_profile = aev_baseline.groupby([aev_baseline.index.hour]).agg(['mean'])
aev_load_profile['hour'] = hour

phev_load_profile = phev_baseline.groupby([phev_baseline.index.hour]).agg(['mean'])
phev_load_profile['hour'] = hour

cchp_load_profile = cchp_baseline.groupby([cchp_baseline.index.hour]).agg(['mean'])
cchp_load_profile['hour'] = hour

ebike_load_profile = ebike_baseline.groupby([ebike_baseline.index.hour]).agg(['mean'])
ebike_load_profile['hour'] = hour


# PLOT 2: Load Profiles
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
fig.suptitle('Average Load Shape Before and After End-Use Instalation')

# subplot 1: AEV
aev_load_profile.plot.line(x='hour', 
                           y=['reporting_observed','counterfactual_usage'],
                           label = ['Observed','Counterfactual'],
                           ax = axes[0,0],
                           title = 'AEV')
plt.xticks(np.arange(0, 25, 3))

# subplot 2: PHEV
phev_load_profile.plot.line(x='hour', 
                            y=['reporting_observed','counterfactual_usage'],
                            label = ['Observed','Counterfactual'],
                            ax = axes[0,1],
                            title = 'PHEV')
plt.xticks(np.arange(0, 25, 3))

# subplot 3: CCHP
cchp_load_profile.plot.line(x='hour', 
                            y=['reporting_observed','counterfactual_usage'],
                            label = ['Observed','Counterfactual'],
                            ax = axes[1,0],
                            title = 'CCHP')
plt.xticks(np.arange(0, 25, 3))


# subplot 4: eBike
ebike_load_profile.plot.line(x='hour', 
                             y=['reporting_observed','counterfactual_usage'],
                             label = ['Observed','Counterfactual'],
                             ax = axes[1,1],
                             title = 'eBike')
plt.xticks(np.arange(0, 25, 3))


### Temperature to filter load profiles:
avg_daily_temp = load_growth.groupby([load_growth.index.date]).agg(['mean'])['tempF']


# 15 coldest days
coldest_days = avg_daily_temp.nsmallest(15,'mean').index
load_growth_cold_days = load_growth.loc[(load_growth.index.floor('D').isin(coldest_days)),:]
load_growth_cold_load_profile = load_growth_cold_days.groupby([load_growth_cold_days.index.hour]).agg(['mean'])
load_growth_cold_load_profile['hour'] = hour


# PLOT 3: 15 Coldest Days
load_growth_cold_load_profile.plot.line(x='hour', y=['AEV','CCHP','PHEV','eBike'],
                                        label = ['AEV','CCHP','PHEV','eBike'],
                                        color = sns.xkcd_palette(colors))
plt.ylabel('kW')
plt.title('Load Profiles during 15 Coldest Days')
plt.xticks(np.arange(0, 25, 3))
plt.legend(loc="upper center")

# 15 warmest days
warmest_days = avg_daily_temp.nlargest(15,'mean').index
load_growth_warm_days = load_growth.loc[(load_growth.index.floor('D').isin(warmest_days)),:]
load_growth_warm_load_profile = load_growth_warm_days.groupby([load_growth_warm_days.index.hour]).agg(['mean'])
load_growth_warm_load_profile['hour'] = hour


# PLOT 4: 15 Warmest Days
load_growth_warm_load_profile.plot.line(x='hour', y=['AEV','CCHP','PHEV','eBike'],
                                        label = ['AEV','CCHP','PHEV','eBike'],
                                        color = sns.xkcd_palette(colors))
plt.ylabel('kW')
plt.title('Load Profiles during 15 Warmest Days')
plt.xticks(np.arange(0, 25, 3))
plt.legend(loc="upper center")



# PLOT 5: Temperature Changepoints... this is very slow (yawn!)
# daily_meter_data = eemeter.meter_data_from_csv('data/cchp_daily.csv', freq = 'daily')

# cchp_meter_data, warnings = eemeter.get_reporting_data(
#         daily_meter_data, start=install_end, max_days=365
#     )

# ax = eemeter.plot_energy_signature(cchp_meter_data, temperature)
# cchp_daily_model_results.plot(
#     ax=ax, candidate_alpha=0.02, with_candidates=True, temp_range=(-5, 88)
# )


load_growth.to_csv('data/load_growth.csv', sep=',')


