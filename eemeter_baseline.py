#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:14:31 2020
@author: fhall
Project: Final Data Science 2: Effects of Electrification on Load Growth

"""

import datetime
import eemeter
import matplotlib.pyplot as plt
import pandas as pd
import pytz
from statistics import mean

##### FUNCTIONS ##############################################################

def eemeter_baseline_ami(file, temperature, install_start, install_end):

    '''
    This method uses linear regression to create hourly load profile to 
    serve as the counterfactual for the period where the new measure has 
    been installed. The two key parts of this model are the temperature 
    and occupancy binning.
    '''
    
    ami_data = eemeter.meter_data_from_csv(file, freq = 'hourly')
    
    # get meter data suitable for fitting a baseline model
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        ami_data, end=install_start, max_days=700
    )
    
    # create design matrix for occupancy and segmentation
    preliminary_design_matrix = (
        eemeter.create_caltrack_hourly_preliminary_design_matrix(
            baseline_meter_data, temperature,
            )
        )
    
    # build 12 monthly models 
    segmentation = eemeter.segment_time_series(
        preliminary_design_matrix.index,
        'three_month_weighted'
        )
    
    # assign an occupancy status to each hour of the week (0-167)
    occupancy_lookup = eemeter.estimate_hour_of_week_occupancy(
        preliminary_design_matrix,
        segmentation = segmentation,
        )
    
    # assign temperatures to bins
    temperature_bins = eemeter.fit_temperature_bins(
        preliminary_design_matrix,
        segmentation = segmentation,
        )
    
    # build a desgin matrix for each monthly segment
    segmented_design_matrices = (
        eemeter.create_caltrack_hourly_segmented_design_matrices(
            preliminary_design_matrix,
            segmentation,
            occupancy_lookup,
            temperature_bins,
            )
        )
    
    # build a CalTRACK hourly model
    baseline_model = eemeter.fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        temperature_bins)
    
    # get a year of reporting data
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        ami_data, start = install_end, max_days = 365
        )
    
    # compute metered load growth for the year of reporitng period
    metered_growth_dataframe, error_bands = eemeter.metered_savings(
        baseline_model, reporting_meter_data,
        temperature, with_disaggregated = True
        )
    
    metered_growth_dataframe['temp'] = temperature # append temperature
    
    
    # totaled load growth
    additional_load = metered_growth_dataframe.metered_savings.sum()

    
    # metrics
    r_squared_adj_list = []
    rmse_adj_list = []
    
    
    # results in a dict
    model_results = list(baseline_model.json().values())
    results = model_results[6]
    for segment, measures in results.items():
        for measure, value in measures.items():
            if measure == 'r_squared_adj':
                r_squared_adj_list.append(value)
            if measure == 'rmse_adj':
                rmse_adj_list.append(value)
    
    r_squared_adj = mean(r_squared_adj_list)
    rmse_adj = mean(rmse_adj_list)

    metrics = [r_squared_adj, rmse_adj]
    
    # Return Section
    return metered_growth_dataframe, metrics, baseline_model
    


def eemeter_baseline_daily(file, temperature, install_start, install_end):
        
    
    
    
    
    
    
    
    
    
    