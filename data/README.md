# Data Folder

This folder contains compact aggregate data used by the public Streamlit app.

## Included Public Data

```text
reference_table_sevo_all_v2_no_n2o_by_ET.csv
reference_table_sevo_all_v1_by_ET.csv
dsa_templates_v1/
des_reference_v4_all_0p1/reference_table_des_all_v4_0p1_no_n2o_by_ET.csv
des_dsa_templates_v4_all_0p1/
```

These files are sufficient for the public app to show reference values, trend plots, DSA templates, and DSA-derived Fourier spectra.

## Local-Only Data

The following working data are intentionally excluded from git:

```text
raw_vitaldb_cache/
reference_rows*.csv
*/reference_rows*.csv
*case_level*.csv
*row_level*.csv
```

If these files are present locally, the development app can show representative raw EEG windows. They are not required for the public aggregate reference app.

## Raw EEG

Raw EEG cache files are not distributed in this repository. They should be regenerated or obtained separately according to the relevant VitalDB data-use terms and local research governance.
