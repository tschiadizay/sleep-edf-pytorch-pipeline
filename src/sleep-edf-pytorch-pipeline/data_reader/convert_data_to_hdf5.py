"""Convert SleepEDF[1, 2] dataset into .hdf5 and .fif files.

Convert the Sleep-EDF dataset from Physionet[1, 2] for each study type into a
.hdf5 file containing all (callibrated by mne.read_raw-edf)
psg data with annotations and one .fif file containing meta data for the whole
study.

Assumptions:
    Assumes that the original dataset downloaded from [1, 2] with its original
    file structure is in src_path.

Sources:
    [1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a
    sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the
    EEG. IEEE-BME 47(9):1185-1194 (2000). https://ieeexplore.ieee.org/document/867928/

    [2] Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C.,
    Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform
    for biomedical research. Nature Health.
    https://doi.org/10.1038/s44360-026-00096-z. Available from:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC13641116/

"""

import logging
import warnings
from pathlib import Path
from typing import Literal

import h5py
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

ENCODING: dict[str, int] = {
    "Sleep stage ?": 0,
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 5,
    "Sleep stage R": 6,
    "Movement time": 7,
}

class ConvertHDF5Processor:
    """Main Class to convert the Sleep-EDF dataset to hdf5 and fif files.

    Class has two main functions: read_metadata and write_files which need
    to be run in this order after constructing an object. In read_metadata,
    the self.df object is created and filled with all metadata relating to
    an individuall session (i.e. age of subject or lights_off_time).
    In write_files, the self.df file is written as a .csv file, the metadata
    which is not session/subject related like channel_names or sample_frequency
    is stored as a .fif object by reading the mne info object of recordings.
    Finally, all psg recordings and their annotation are also written into
    a singlehdf5 file.

    Object attributes:
        folder_path (Path): Path of the main folder where the dataset is
            already downloaded.
        study_type (Literal["cassette", "telemetry"]): String which tells
            whether the "cassette" or "telemetry" dataset should be read.
        files_location (Path): Path of the .edf files of the study.
        df (df.DataFrame): Panda Dataframe containing for all night recordings
            related metadata like age, sex, scorer.

    """

    def __init__(self, folder_path: str | Path,
                 study_type: Literal["cassette", "telemetry"]):
        """Construct a ConvertHDF5Processor without reading/writing files yet.

        Args:
            folder_path (str | Path): The path of the downloaded Sleep-EDF
                Dataset i.e. 'physionet.org/files/sleep-edfx/1.0.0.'.
            study_type ("cassette", "telemetry"): Study_type which should be
                converted.

        """
        self.folder_path = Path(folder_path)
        self.study_type = study_type
        self.files_location = self.folder_path / ("sleep-" + self.study_type)
        self.df = None

    def read_metadata_from_xls_into_df(self):
        """Create and fill self.df with information read from xls.

        Creates a dataframe in self.df containing for each subject: subject,
        age, sex, night, lights_off_time, (and is_placebo_session for
        self.study_type='telemtry').

        """
        if self.study_type == "cassette":
            xls_path = self.folder_path / "SC-subjects.xls"
            df = pd.read_excel(xls_path, header=[0])
            df = df.rename(
                columns={
                    "subject": "subject",
                    "age": "age",
                    "sex (F=1)": "sex",
                    "night": "night",
                    "LightsOff": "lights_off_time",
                }
            )
            df["sex"] = df["sex"].replace({1: "F", 2: "M"}) # "F=1" acc. to .csv
            df = df[
                ["subject", "age", "sex", "night", "lights_off_time"]
            ]
        elif self.study_type == "telemetry":
            xls_path = self.folder_path / "ST-subjects.xls"
            df = pd.read_excel(xls_path, header=[1])
            df = df.rename(
                columns={
                    "Nr": "subject",
                    "Age": "age",
                    "M1/F2": "sex",
                    "night nr": "placebo_night",
                    "lights off": "placebo_lights_off_time",
                    "night nr.1": "temezepam_night",
                    "lights off.1": "temezepam_lights_off_time",
                }
            )
            # create two subdfs from the original df for the placebo and
            # temezepamgroup and then concatenate them together into a single
            # df with a is_placebo_group flag
            placebo_group = df[
                [
                    "subject",
                    "age",
                    "sex",
                    "placebo_night",
                    "placebo_lights_off_time",
                ]
            ]
            temezepam_group = df[
                [
                    "subject",
                    "age",
                    "sex",
                    "temezepam_night",
                    "temezepam_lights_off_time",
                ]
            ]
            placebo_group = placebo_group.rename(
                columns={
                    "placebo_night": "night",
                    "placebo_lights_off_time": "lights_off_time",
                }
            )
            temezepam_group = temezepam_group.rename(
                columns={
                    "temezepam_night": "night",
                    "temezepam_lights_off_time": "lights_off_time",
                }
            )

            df = pd.concat([placebo_group, temezepam_group], ignore_index=True)

            df["is_placebo_session"] = np.concatenate(
                (
                    np.ones(len(placebo_group), dtype=bool),
                    np.zeros(len(temezepam_group), dtype=bool),
                )
            )

            df["sex"] = df["sex"].replace(
                {1: "M", 2: "F"}) # "M1/F2" acc. to .csv

        df["subject"] = [i.zfill(2) for i in df["subject"].astype(str)]
        df["night"] = df["night"].astype(str)
        df = df.sort_values(by=["subject", "night"])
        df.index = df["subject"] + "-" + df["night"]
        df["lights_off_time"] = df["lights_off_time"].apply(
            lambda x: x.isoformat()
        )
        return df

    def add_metadata_from_files_into_df(self):
        """Add metadata read from files and filenames into self.df.

        Add in the dataframe self.df for each subject random_seventh_letter
        from the filename, scorer of the session, the filename of the PSG.edf
        and Hypnogram.edf and measurement_date of the session by reading the
        filename and the mne.io.read_raw_edf.info object

        """
        assert self.df is not None
        metadata = []
        logging.info("Start: add_metadata_from_files_into_df")
        for hypnogram_file in tqdm(self.files_location.glob("*Hypnogram.edf")):
            hypnogram_file_name = hypnogram_file.name
            subject = hypnogram_file_name[3:5]
            night = hypnogram_file_name[5]
            random_seventh_letter = hypnogram_file_name[6]
            scorer = hypnogram_file_name[7]
            filename_psg = hypnogram_file_name[:7] + "0-PSG.edf"
            filename_hypnogram = hypnogram_file_name

            infos = {
                "subject": subject,
                "night": night,
                "random_seventh_letter": random_seventh_letter,
                "scorer": scorer,
                "filename_psg": filename_psg,
                "filename_hypnogram": filename_hypnogram,
            }

            infos["measurement_date"] = mne.io.read_raw_edf(
                self.files_location / filename_psg, verbose="ERROR"
                ).info["meas_date"]
            metadata.append(infos)

        metadata = pd.DataFrame.from_records(metadata)
        metadata.index = metadata["subject"] + "-" + metadata["night"]
        metadata["measurement_date"] = metadata["measurement_date"].apply(
            lambda x: x.isoformat()
        )
        self.df = pd.merge(self.df, metadata, on=["subject", "night"])
        self.df.index = self.df["subject"] + "-" + self.df["night"]
        self.df.index.name = "INDEX"

    def write_as_hdf5(self, path : str | Path | None = None):
        """Read PSG recordings and annotations and store inside .hdf5 file.

        Writes for each session a group and inside: a dataset for the PSG data
        and a dataset for the Hypnogram data into a hdf5 file. In Addition,
        store in the group session-releated metadata the unique key identifying
        the session (subject_id and night_recording).

        Args:
            path (str | Path | None, optional): Path of the resulting .hdf5
                file. If no path is given, default path is used.
                Defaults to None.

        Results:
            The file f f"Sleep-{self.study_type}.hdf5 will
                contain the whole PSG and Hypnogram data of the study.
            f["/" + str(row["subject"]) + "-" +  str(row["night"])] will
                return the group of the session.
            The datasets inside can be accessed with "PSG_DATA" or
                "HYPNOGRAM_DATA".

        """
        assert self.df is not None
        if path is None:
            path = f"Sleep-{self.study_type}.hdf5"
            warnings.warn(f"No Path given. Using default Path: {path}",
                          stacklevel=2)
        with h5py.File(path, "w") as f:
            logging.info("Start: write_as_hdf5")
            for row in tqdm(self.df.to_dict("records")):
                raw = mne.io.read_raw_edf(
                    self.files_location / row["filename_psg"],
                    verbose="ERROR",
                    preload=True,
                )
                psg_data = raw.get_data().astype("float32")  # type: ignore

                annotations = mne.read_annotations(
                    self.files_location / row["filename_hypnogram"]
                )
                hypnogram_data = np.array(
                    (
                        annotations.onset,
                        annotations.duration,
                        np.array(
                            [
                                ENCODING[epoch]
                                for epoch in annotations.description
                            ],
                            dtype=np.int8,
                        ),
                    )
                )

                grp = f.create_group(
                    str(row["subject"]) + "-" + str(row["night"])
                )
                grp.create_dataset(
                    "PSG_DATA", data=psg_data, compression="gzip", shuffle=True
                )
                grp.create_dataset(
                    "HYPNOGRAM_DATA",
                    data=hypnogram_data,
                    compression="gzip",
                    shuffle=True,
                )

    def store_metadata_as_fif(self, path : str | Path | None = None):
        """Store all common fields from mne.info objects into a .fif file.

        Store for the whole study the stored metadata (which is not subject
        and session related and therefore for all sessions relevant) into
        a .fif file. It was tested in a separate script that the metadata
        stored in this .fif file was the same for all session, otherwise it was
        stored in the sessions group.

        Args:
            path (str | Path | None, optional): Path of the resulting .fif
                file. If no path is given, default path is used.
                Defaults to None.

        """
        if path is None:
            path = f"{self.study_type}-metadata.fif"
            warnings.warn(f"No Path given. Using default Path: {path}",
                stacklevel=2)
        file_name = next(self.files_location.glob("*PSG.edf"))
        raw = mne.io.read_raw_edf(file_name, verbose="ERROR")

        # Set attributes which are not universal to None
        # (they are stored in the sessions group data anyway)
        raw.info["subject_info"] = None
        raw.info.set_meas_date(None)
        mne.io.write_info(path, raw.info, overwrite=True, verbose="error")

    def store_session_metadata_as_csv(self, path : str | Path | None = None):
        """Store the SessionMetadata stored in self.df into path as .csv.

        Args:
            path (str | Path | None, optional): Path of the resulting .csv
                file. If no path is given, default path is used.
                Defaults to None.

        """
        assert self.df is not None
        if path is None:
            path = f"{self.study_type}-DATA.csv"
            warnings.warn(f"No Path given. Using default Path: {path}",
                stacklevel=2)
        self.df.to_csv(path, index=True)

    def read_metadata(self):
        """Encapsulate all methods for reading metadata into self.df."""
        self.df = self.read_metadata_from_xls_into_df()
        self.add_metadata_from_files_into_df()
        logger.info("Reading Metadata finished.")

    def write_files(self, fif_path : str | Path | None = None,
                    hdf5_path : str | Path | None = None,
                    save_session_metadata_as_csv : bool | None = True,
                    csv_path : str | Path | None = None):
        """Encapsulate all methods which actually write and create files.

        Args:
            fif_path (str | Path | None, optional): Path of the resulting .fif
                file. If no path is given, default path is used.
                Defaults to None.
            hdf5_path (str | Path | None, optional): Path of the resulting
                .hdf5 file. If no path is given, default path is used.
                Defaults to None.
            save_session_metadata_as_csv (bool | None, optional): Whether
                session_related metadata should be stored. Defaults to True.
            csv_path (str | Path | None, optional): Path of the resulting
                .csv file. If no path is given, default path is used.
                Defaults to None.

        """
        if save_session_metadata_as_csv:
            self.store_session_metadata_as_csv(csv_path)
        self.store_metadata_as_fif(fif_path)
        self.write_as_hdf5(hdf5_path)

if __name__ == "__main__":
    import utils.logging_setup
    utils.logging_setup.set_up_logging()
    logger = logging.getLogger(__name__)

    logger.info("Started %s", __name__)

    # Call to create for each study_type the .hdf5 file containing all PSG and
    # Hypnogram data and one .fif file containing all metadata of the study.
    base_dir = Path(
        "" # TODO
    )
    src_path = base_dir / "physionet.org/files/sleep-edfx/1.0.0"
    out_path = base_dir / "SleepEDFConverted"
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Telemtry Dataset Processing started.")

    # Telemetry Dataset
    telemetry_processor = ConvertHDF5Processor(src_path, "telemetry")
    telemetry_processor.read_metadata()
    telemetry_processor.write_files(
        fif_path=out_path/f"{telemetry_processor.study_type}-metadata.fif",
        hdf5_path=out_path/f"Sleep-{telemetry_processor.study_type}.hdf5",
        save_session_metadata_as_csv=True,
        csv_path=out_path/f"{telemetry_processor.study_type}-DATA.csv",
    )
    logger.info("Telemtry Dataset Processing finished.")

    logger.info("Cassette Dataset Processing started.")

    # Cassette Dataset
    cassette_processor = ConvertHDF5Processor(src_path, "cassette")
    cassette_processor.read_metadata()
    cassette_processor.write_files(
        fif_path=out_path/f"{cassette_processor.study_type}-metadata.fif",
        hdf5_path=out_path/f"Sleep-{cassette_processor.study_type}.hdf5",
        save_session_metadata_as_csv=True,
        csv_path=out_path/f"{cassette_processor.study_type}-DATA.csv",
    )
    logger.info("Cassette Dataset Processing finished.")
