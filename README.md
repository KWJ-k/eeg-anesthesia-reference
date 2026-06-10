# EEG Anesthesia Reference

Interactive Streamlit app for exploring age- and anesthetic-concentration-specific EEG reference values derived from VitalDB.

The current app focuses on sevoflurane and desflurane reference cells. It shows expected BIS, SEF, EEG band power, age trends, concentration trends, precomputed DSA templates, and DSA-derived Fourier spectra.

## Features

- Sevoflurane and desflurane reference lookup
- Age-bin input
- End-tidal concentration mode
- Age-adjusted MAC mode
- BIS, SEF, alpha power, and cell case count summaries
- Age trend slope with weighted linear trend
- Band-power bar chart
- Age and concentration trend plots
- Precomputed raw VitalDB DSA templates
- DSA-derived Fourier spectrum with reference-table SEF marker
- Optional local representative raw EEG display when local raw EEG cache files are available

## Data Policy

This public repository is intended to include compact aggregate reference tables and precomputed DSA templates only.

Raw EEG cache files are intentionally excluded from git:

```text
data/raw_vitaldb_cache/
```

Window-level and case-level working tables are also excluded from the public repository. They may be present on a local development machine and can enable representative raw EEG display, but they are not required to run the public app.

## Included App Data

The app expects these compact files/directories under `data/`:

```text
data/reference_table_sevo_all_v2_no_n2o_by_ET.csv
data/reference_table_sevo_all_v1_by_ET.csv
data/dsa_templates_v1/
data/des_reference_v4_all_0p1/reference_table_des_all_v4_0p1_no_n2o_by_ET.csv
data/des_dsa_templates_v4_all_0p1/
```

## Installation

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app/streamlit_app_trend_v1.py
```

Then open:

```text
http://localhost:8501
```

If another Streamlit app is already using port 8501, Streamlit may open a different port such as 8502.

## Project Layout

```text
eeg-reference-app/
  app/
    streamlit_app_trend_v1.py
  data/
    README.md
  docs/
  figures/
  notebooks/
  scripts/
  src/
    dsa_reference.py
    plotting.py
    raw_eeg_reference.py
    reference_lookup.py
    vitaldb_raw.py
  requirements.txt
```

## Scientific Note

This app is a research tool for exploring EEG reference patterns. It is not a clinical decision-support system and should not be used as a substitute for clinical judgment or validated monitoring devices.

## Status

Research prototype. Interfaces and reference datasets may change as the analysis pipeline is refined.
