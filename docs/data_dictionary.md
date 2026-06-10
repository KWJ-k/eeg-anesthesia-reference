# Data Dictionary

This document describes the main reference table columns.

## Identification

- `agent`: inhaled agent, currently `sevo`.
- `reference_sex`: reference grouping sex, currently usually `all`.
- `age_bin`: 5-year age bin such as `60-64`.
- `target_agent`: rounded end-tidal sevoflurane concentration target.

## Sample Size

- `n_cases_clean`: number of unique cases in the reference cell after QC.
- `n_windows_clean`: number of accepted windows in the reference cell.
- `n_female`: number of female windows/cases contributing to the cell.
- `n_male`: number of male windows/cases contributing to the cell.
- `usable_cell`: `True` when `n_cases_clean` meets the threshold.

## Concentration

- `mean_agent`: mean end-tidal sevoflurane concentration in the accepted windows.
- `sd_agent`: standard deviation of end-tidal sevoflurane concentration.
- `mean_mac`: mean Primus MAC value.

## Processed EEG

- `mean_bis`: mean BIS value in the accepted windows.
- `sd_bis`: standard deviation of BIS values.
- `mean_sef`: mean spectral edge frequency.
- `sd_sef`: standard deviation of SEF.
- `mean_sqi`: mean BIS signal quality index.

## Frequency Features

All power values are in dB and are based on multitaper spectral estimates.

- `delta_power_db`: average delta band power.
- `theta_power_db`: average theta band power.
- `alpha_power_db`: average alpha band power.
- `beta_power_db`: average beta band power.
- `alpha_peak_freq`: median alpha peak frequency.
- `alpha_peak_power_db`: average alpha peak power.
- `alpha_delta_diff_db`: alpha power minus delta power.

## N2O Fields In v2

- `mean_fen2o`: mean expired nitrous oxide concentration.
- `mean_fin2o`: mean inspired nitrous oxide concentration.
- `max_fen2o`: maximum expired nitrous oxide concentration.
- `max_fin2o`: maximum inspired nitrous oxide concentration.
