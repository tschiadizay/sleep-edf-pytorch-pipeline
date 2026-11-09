# sleep-edf-pytorch-pipeline

Scripts to process the Sleep-EDF Expanded dataset [1, 2] into aligned, filtered .hdf5 files and load them into PyTorch with a stratified, subject-grouped K-fold split. The three scripts must be run in order. `{study}` below is "telemetry" or "cassette".

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
cd <base_dir>
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
```

The download creates "<base_dir>/physionet.org/files/sleep-edfx/1.0.0/" with the needed files and file structure.

## Running

Run all scripts as modules from "src/sleep-edf-pytorch-pipeline/" (because relative paths are used). Each script has a "base_dir = Path("")  # TODO" in its ```__main__``` block that you must set first with `<base_dir>`:

### 1. Convert .edf files to .hdf5 file

```bash
python -m data_reader.convert_data_to_hdf5
```

Converts both studies. For each, it writes "Sleep-{study}.hdf5", "{study}-metadata.fif" and "{study}-DATA.csv" to "SleepEDFConverted/". The signals are unfiltered and the hypnogram is still in its original continuous format (onset, duration, label). The only processing is the calibration of the digitized signal into physical values by "mne.io.read_raw_edf()", the conversion to float32, and the encoding of the stage names as integer labels 0 to 7.

### 2. Sort and filter data:

```bash
python -m data_reader.sort_and_filter_data
```

Band-pass filters the channels, aligns the signal with the hypnogram, cuts it into 30 s epochs, trims Wake epochs around the sleep period, and removes and merges labels. Writes "Processed_Sleep-{study}.hdf5" and "Processed_Sleep-{study}-DATA.csv". Channels, filters, epoch length, trimming and the label mapping are set in the "Config(...)" in the ```__main__``` block. With the default mapping the labels are 0 Wake, 1 N1, 2 N2, 3 N3, 4 REM.

### 3. Load into PyTorch:

"load_data_to_pytorch.py" provides the "SleepEDFDataset" class and a stratified, subject-grouped K-fold split. The ```__main__``` block of the file contains an exemplary import into PyTorch using these methods: it creates the folds, builds a train and a validation "SleepEDFDataset" for each fold, and wraps them in "DataLoader"s. Set "base_dir" there and run it with:

```bash
python -m data_reader.load_data_to_pytorch
```

Each item of the dataset is one 30 s epoch of shape "(n_channels, 3000)" (float32) with its label (long).

## References

[1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000). https://ieeexplore.ieee.org/document/867928/

[2] Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z
