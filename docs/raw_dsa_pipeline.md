# Raw VitalDB DSA Pipeline

## Goal

MVP v1 uses precomputed broad band features. v2 should build true DSA templates
from VitalDB raw EEG waveform data.

The intended pipeline is:

1. Use the stable window table:
   `data/reference_rows_sevo_all_v2_no_n2o_qc.csv`
2. For each row, download `BIS/EEG1_WAV` from VitalDB using `case_id`.
3. Slice the raw waveform by `window_start_sec` and `window_end_sec`.
4. Compute short-window FFT/PSD.
5. Aggregate into 1 Hz frequency bins from 0-40 Hz.
6. Median-average DSA matrices within each `age_bin` and `target_agent` cell.
7. Save one `.npz` template per cell plus `template_index.csv`.

## Why Not Use The MVP Band Powers

Delta, theta, alpha, and beta powers are useful summary features, but they are
too broad for a DSA reference. They cannot show narrow alpha peak shifts,
spectral edge behavior, or frequency-specific age changes. The DSA layer must
come from raw EEG Fourier analysis, not from reconstructed band summaries.

## Current Implementation

Core modules:

```text
src/vitaldb_raw.py
src/eeg_spectrum.py
scripts/build_dsa_reference_from_vitaldb.py
```

Development command:

```bash
python scripts/build_dsa_reference_from_vitaldb.py --max-cells 2 --max-windows-per-cell 3
```

Build one specific app cell:

```bash
python scripts/build_dsa_reference_from_vitaldb.py --age-bin 45-49 --target-agent 2.4 --max-windows-per-cell 5 --require-usable
```

Full build command:

```bash
python scripts/build_dsa_reference_from_vitaldb.py --require-usable --all-windows --overwrite
```

## Output Format

Each `.npz` file stores:

- `time_sec`
- `freq_bin_start_hz`
- `freq_bin_end_hz`
- `freq_center_hz`
- `power_db`
- `age_bin`
- `target_agent`
- `n_windows_requested`
- `n_errors`
- `valid_epoch_fraction`

The matrix shape is:

```text
frequency_bins x time_bins
```

## Notes

- Default EEG track: `BIS/EEG1_WAV`
- Default sample rate: 128 Hz
- Default FFT epoch: 4 sec
- Default step: 1 sec
- Default frequency bins: 0-1, 1-2, ..., 39-40 Hz
- Raw case/track downloads are cached in `data/raw_vitaldb_cache/`.
- Window selection is sorted by high `window_quality_score`, high `mean_sqi`,
  low `artifact_ratio`, low `agent_bin_error`, then `case_id`.

The app should only display v2 DSA templates after this raw-data build step has
created real templates.

## DES Expansion

DES does not currently have a prebuilt reference row table in this repository.
Start by building a DES row/table dataset from VitalDB:

```bash
python scripts/build_des_reference_dataset.py --max-cases 50
```

Full DES builds use:

- agent track: `Primus/EXP_DES`
- inspired DES track: `Primus/INSP_DES` if needed later
- MAC track: `Primus/MAC`
- EEG track: `BIS/EEG1_WAV`
- BIS/SEF/SQI tracks from `BIS/*`

The first DES builder uses stable 120-second windows, excludes N2O by
`Primus/FEN2O` and `Primus/FIN2O`, computes FFT-based 1 Hz EEG spectra, and
summarizes the same band features as the sevo reference table.
