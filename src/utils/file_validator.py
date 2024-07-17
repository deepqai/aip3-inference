import logging
import os
import stat
from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from constants.preprocess import IMG_FOLDER


class FileValidator(ABC):
    def __init__(self, file_type, datadir):
        self.datadir = datadir
        self.img_folder_path = os.path.join(self.datadir, IMG_FOLDER)

        self.series_type = "2d"
        if file_type == "dicom":
            self.series_type = "2.5d"

        self._folder_structure_depth = {
            "2.5d": 2,
        }

    def _is_file_readable(self, file_path):
        """
        Checks if the file specified by 'file_path' exists.
        Optionally checks if we have read permission with the 'check_access' flag.

        file_path: (str) the file to be be checked if it exists
        """
        if not os.path.isfile(file_path):
            return False

        file_path = Path(file_path)
        st = file_path.stat()

        return bool(st.st_mode & stat.S_IRGRP)

    @abstractmethod
    def _is_file_type_valid(self, file_path):
        pass

    def _check_file_valid(self, frame):
        """
        Checks if the file_names in 'df' are all workable images in the image folder.
        Logs in an error file if there exists erroneous files.
        Modifies the dataframe (subset of original) with only workable images

        df: (pandas.DataFrame) a DataFrame object representing the csv file being read (contains all info, not a subset)

        Returns a dataframe with the erroneous rows removed.

        return df.reset_index(drop=True)
        """
        file_path = os.path.join(self.img_folder_path, frame.name)

        if not self._is_file_readable(file_path):
            frame["file_valid"] = "Unreadable"
            frame["file_width"] = None
            frame["file_height"] = None
            return frame

        file_valid, file_info = self._is_file_type_valid(file_path)
        (
            file_width,
            file_height,
        ) = file_info

        frame["file_valid"] = file_valid
        frame["file_width"] = file_width
        frame["file_height"] = file_height

        return frame

    def _check_series_type_valid(self, group_df):
        folder_structure_depth = self._folder_structure_depth[self.series_type]

        assert all(
            group_df.apply(
                lambda frame: len(frame.name.split("/")) >= folder_structure_depth
            )
        ), f"The provided dataset: {self.img_folder_path} is not valid for {self.series_type}"

    def _write_file_error_log(self, df):
        error_df = df[df["file_valid"] != "Valid"].drop_duplicates("file_name")

        if not error_df.empty:
            warning_log = error_df.apply(
                lambda frame: f"{frame['file_name']} can not be used due to {frame['file_valid']} error.",
                axis=1,
            )
            warning_message = "FOLLOWING STUDIES ARE NOT VALID\n"
            warning_message += "\n".join(warning_log.values)

            print(warning_message)

    def validate(self, df):
        group_df = df.groupby("file_name")

        if self.series_type == "2.5d":
            self._check_series_type_valid(group_df)

        df = group_df.apply(lambda frame: self._check_file_valid(frame))

        self._write_file_error_log(df)

        return df[df["file_valid"] == "Valid"].copy().reset_index(drop=True)


class ImageValidator(FileValidator):
    def _is_file_type_valid(self, file_path):
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            height, width, channel = img.shape
            if height <= 1 or width <= 1 or channel != 3:
                return "Non_Image_File", (None, None)
        except Exception:
            return "Non_Image_File", (None, None)
        return "Valid", (width, height)
