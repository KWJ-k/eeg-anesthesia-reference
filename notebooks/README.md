# Notebooks

Recommended notebook organization:

```text
01_build_reference_dataset.ipynb
02_analysis_age_bis_sef.ipynb
03_colab_mvp_app.ipynb
```

## 01 Build Reference Dataset

Purpose:

- Use VitalDB.
- Select stable EEG windows.
- Compute multitaper spectrum features.
- Build feature reference tables.

## 02 Analysis Age BIS SEF

Purpose:

- Explore age effects near MAC 0.8-1.2.
- Compare BIS, SEF, and EEG band power.
- Generate correlation and regression figures.

## 03 Colab MVP App

Purpose:

- Load the final reference CSV.
- Provide interactive lookup by age and ET sevo or MAC.
- Prototype the UI before moving logic into Streamlit.
