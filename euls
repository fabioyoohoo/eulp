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
import pandas as pd
import pytz
import seaborn as sns
from statistics import mean
# from matplotlib.pyplot import figure
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
aev_daily_baseline, aev_growth, aev_daily_model_results = eemeter_baseline_daily('data/aev_daily.csv', temperature, install_start, install_end)
phev_daily_baseline, phev_growth, phev_daily_model_results = eemeter_baseline_daily('data/phev_daily.csv', temperature, install_start, install_end)
cchp_daily_baseline, cchp_growth, cchp_daily_model_results = eemeter_baseline_daily('data/cchp_daily.csv', temperature, install_start, install_end)
ebike_daily_baseline, ebike_growth, ebike_daily_model_results = eemeter_baseline_daily('data/ebike_daily.csv', temperature, install_start, install_end)



# combine load growth
load_growth = pd.concat([aev_baseline['metered_savings'],
                     phev_baseline['metered_savings'],
                     cchp_baseline['metered_savings'],
                     ebike_baseline['metered_savings'],
                     ebike_baseline['temp']], 
                    axis=1)
load_growth.columns = ['AEV','PHEV','CCHP','eBike','tempF']



##############################################################################
#### VISUALIZATION ###########################################################
##############################################################################

temp_bins = load_growth.tempF//20
temp_labels = ['below 0','0 to 19','20 to 39','40 to 59','60 to 79', 'above 80']

# determine mean usage by temperature bin
load_growth_grouped = load_growth.groupby(temp_bins).agg(['mean'])*-1 # flips sign
load_growth_grouped['temp_bin'] = temp_labels
colors = ["windows blue", "amber", "greyish", "faded green"]

# grouped savings by
load_growth_grouped.plot.bar(x='temp_bin',y=['AEV','CCHP','PHEV','eBike'], 
                       label = ['AEV','CCHP','PHEV','eBike'],
                       color = sns.xkcd_palette(colors))
plt.xticks(rotation=0)
plt.xlabel('temperature bins (F)')
plt.ylabel('mean kWh increase')


# determine total usage by temperature bin
load_growth_grouped = load_growth.groupby(temp_bins).agg(['sum'])*-1 # flips sign
load_growth_grouped['temp_bin'] = temp_labels
colors = ["windows blue", "amber", "greyish", "faded green"]

# grouped savings by
load_growth_grouped.plot.bar(x='temp_bin',y=['AEV','CCHP','PHEV','eBike'], 
                       label = ['AEV','CCHP','PHEV','eBike'],
                       color = sns.xkcd_palette(colors))
plt.xticks(rotation=0)
plt.xlabel('temperature bins (F)')
plt.ylabel('mean kWh increase')


### LOAD PROFILE
hour = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
aev_load_profile = aev_baseline.groupby([aev_baseline.index.hour]).agg(['mean'])
aev_load_profile['hour'] = hour
aev_load_profile.plot.line(x='hour', y=['reporting_observed','counterfactual_usage'])

phev_load_profile = phev_baseline.groupby([phev_baseline.index.hour]).agg(['mean'])
phev_load_profile['hour'] = hour
phev_load_profile.plot.line(x='hour', y=['reporting_observed','counterfactual_usage'])

cchp_load_profile = cchp_baseline.groupby([cchp_baseline.index.hour]).agg(['mean'])
cchp_load_profile['hour'] = hour
cchp_load_profile.plot.line(x='hour', y=['reporting_observed','counterfactual_usage'])

ebike_load_profile = ebike_baseline.groupby([ebike_baseline.index.hour]).agg(['mean'])
ebike_load_profile['hour'] = hour
ebike_load_profile.plot.line(x='hour', y=['reporting_observed','counterfactual_usage'])


avg_daily_temp = load_growth.groupby([load_growth.index.date]).agg(['mean'])['tempF']


# 15 coldest days
coldest_days = avg_daily_temp.nsmallest(15,'mean').index
cchp_cold_days = cchp_baseline.loc[(cchp_baseline.index.floor('D').isin(coldest_days)),:]

cchp_cold_load_proflie = cchp_cold_days.groupby([cchp_cold_days.index.hour]).agg(['mean'])
cchp_cold_load_proflie['hour'] = hour
cchp_cold_load_proflie.plot.line(x='hour', y=['reporting_observed','counterfactual_usage'])


# 15 warmest days
warmest_days = avg_daily_temp.nlargest(15,'mean').index
cchp_cold_days = cchp_baseline.loc[(cchp_baseline.index.floor('D').isin(warmest_days)),:]






# ### get test datasets
# meter_data, temperature_data, sample_metadata = (
#     eemeter.load_sample("il-electricity-cdd-hdd-billing_monthly")
# )

# aev_data = eemeter.meter_data_from_csv('data/cchp_ami.csv', freq = 'hourly')
# aev_hourly_matrix = eemeter.create_caltrack_hourly_preliminary_design_matrix(aev_data, temperature)

# get meter data suitable for fitting a baseline model
# baseline_meter_data, warnings = eemeter.get_baseline_data(
#     aev_data, end=blackout_start_date, max_days=700
# )

# # create design matrix for occupancy and segmentation
# preliminary_design_matrix = (
#     eemeter.create_caltrack_hourly_preliminary_design_matrix(
#         baseline_meter_data, temperature,
#         )
#     )

# # build 12 monthly models 
# segmentation = eemeter.segment_time_series(
#     preliminary_design_matrix.index,
#     'three_month_weighted'
#     )

# # assign an occupancy status to each hour of the week (0-167)
# occupancy_lookup = eemeter.estimate_hour_of_week_occupancy(
#     preliminary_design_matrix,
#     segmentation = segmentation,
#     )

# # assign temperatures to bins
# temperature_bins = eemeter.fit_temperature_bins(
#     preliminary_design_matrix,
#     segmentation = segmentation,
#     )

# # build a desgin matrix for each monthly segment
# segmented_design_matrices = (
#     eemeter.create_caltrack_hourly_segmented_design_matrices(
#         preliminary_design_matrix,
#         segmentation,
#         occupancy_lookup,
#         temperature_bins,
#         )
#     )

# # build a CalTRACK hourly model
# baseline_model = eemeter.fit_caltrack_hourly_model(
#     segmented_design_matrices,
#     occupancy_lookup,
#     temperature_bins)



# # get a year of reporting data
# reporting_meter_data, warnings = eemeter.get_reporting_data(
#     aev_data, start = blackout_end_date, max_days = 365
#     )

# # compute metered load growth for the year of reporitng period
# metered_growth_dataframe, error_bands = eemeter.metered_savings(
#     baseline_model, reporting_meter_data,
#     temperature, with_disaggregated = True
#     )


# metered_growth_dataframe['temp'] = temperature

# # totaled load growth
# total_aev_load = metered_growth_dataframe.metered_savings.sum()

# aev_before = aev_data[aev_data.index <= blackout_start_date]



# # Model Results
# # model_results = eemeter.CalTRACKHourlyModelResults(
# #     baseline_meter_data, aev_before)


# r_squared_adj_list = []
# rmse_adj_list = []


# # results in a dict
# model_results = list(baseline_model.json().values())
# results = model_results[6]
# for segment, measures in results.items():
#     for measure, value in measures.items():
#         if measure == 'r_squared_adj':
#             r_squared_adj_list.append(value)
#         if measure == 'rmse_adj':
#             rmse_adj_list.append(value)

# r_squared_adj = mean(r_squared_adj_list)
# rmse_adj = mean(rmse_adj_list)



# print(json.dumps(baseline_model.json(), indent = 2))




# ax = eemeter.plot_energy_signature(aev_data, temperature)
# baseline_model.plot(
#     ax = ax, candidate_alpha = 0.002, with_candidate = True,
#     temp_range = (-5, 88))


# # create a design matrix suitable for use with billing data
# baseline_design_matrix = eemeter.create_caltrack_hourly_preliminary_design_matrix(
#     baseline_meter_data, temperature_data,
# )


# # build a CalTRACK model
# baseline_model = eemeter.fit_caltrack_usage_per_day_model(
#     baseline_design_matrix,
# )

# # get a year of reporting period data
# reporting_meter_data, warnings = eemeter.get_reporting_data(
#     meter_data, start=blackout_end_date, max_days=365
# )

# # compute metered savings for the year of the reporting period we've selected
# metered_savings_dataframe, error_bands = eemeter.metered_savings(
#     baseline_model, reporting_meter_data,
#     temperature_data, with_disaggregated=True
# )

# # total metered savings
# total_metered_savings = metered_savings_dataframe.metered_savings.sum()


# data = metered_savings_dataframe[(metered_savings_dataframe.index > '2015-08-01T01:00:00.000000000')]

# # plt.figure(num=None, figsize=(20, 10), dpi=800, facecolor='w', edgecolor='k')
# plt.rcParams["figure.figsize"] = (18,12)
# metered_savings_dataframe.plot(kind = 'line', alpha = .75, linewidth = 7)
# plt.legend(loc='upper left',)
# plt.title("Baselined Counterfactuals")
# plt.xlabel('Hourly Date_times')
# plt.ylabel('kWh')
# plt.show()


# import datetime
# import pytz
# datetime.datetime(2016, 12, 26, 0, 0, tzinfo=pytz.UTC)

# baseline_data, warnings = eemeter.get_baseline_data(data, end=blackout_end_date, max_days=365)
# model_results = eemeter.fit_caltrack_usage_per_day_model(baseline_data)
# ax = eemeter.plot_energy_signature(meter_data, temperature_data)
# model_results.plot(ax=ax, with_candidates=True)
