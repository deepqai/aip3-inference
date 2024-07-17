import json
import logging
import math
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from constants.default import DEFAULT_PRETRAIN_DIR
from constants.preprocess import (
    CLASSES_FILE,
    IMG_FOLDER,
    LABEL_FILE,
    NO_ANNOTATION_LABEL,
)

from .endecode import encode, mask_area_percentage


class Preprocess(ABC):
    def __init__(self, args, file_validator):
        self.task_type = args.task_type
        self.datadir = args.datadir
        self.img_folder_path = os.path.join(self.datadir, IMG_FOLDER)
        self.pretrained_model_json = f"{DEFAULT_PRETRAIN_DIR}/model.json"

        self.label_path = os.path.join(self.datadir, LABEL_FILE)
        self.predict_only = not os.path.isfile(self.label_path)

        self.no_annotation_label = NO_ANNOTATION_LABEL
        self.classes_file_path = os.path.join(self.datadir, CLASSES_FILE)

        self.inference = args.inference
        self.file_validator = file_validator
        self.file_type = args.file_type

        self.val_ratio = args.val_ratio

    def _get_image_list(self):
        abs_datadir = os.path.abspath(self.img_folder_path)

        imglist = []
        for dir_path, _, file_names in os.walk(abs_datadir):
            for file_name in file_names:
                abs_file_path = os.path.join(dir_path, file_name)
                relative_file_path = os.path.relpath(
                    abs_file_path, os.path.join(abs_datadir)
                )
                imglist.append(relative_file_path)

        return imglist

    def _get_class_list_from_df(self, df):
        class_name = df["label"]
        class_name = class_name[class_name != self.no_annotation_label]

        class_list = sorted(class_name.unique())
        class_num = len(class_list)

        if self.task_type == "detection":
            class_list = [self.no_annotation_label] + class_list

        return class_list, class_num

    def _generate_file(self, file_map):
        """
        Generates a dummy csv from image folder.
        Saves it at args.datadir as 'inference.csv' or inference.json.
        """
        imglist = (
            [[img] for img in self._get_image_list()]
            if not file_map
            else sorted(file_map.keys())
        )

        # create dummy dataframe
        df = pd.DataFrame(imglist, columns=["original_name"])
        df["file_name"] = df["original_name"]
        if file_map:
            df.loc[:, "file_name"] = df.loc[:, "file_name"].apply(
                lambda name: file_map[name]
            )

        df = self.file_validator.validate(df)
        self._df_to_file(
            df=df,
            file_name="inference",
            columns=[
                "file_name",
                "original_name",
                "file_width",
                "file_height",
            ],
        )

    def _df_to_file(self, df, file_name, columns=None):
        file_name = file_name + ".csv"

        if columns is None:
            columns = self.output_csv_columns

        df = df.loc[:, columns]
        df.to_csv(
            os.path.join(self.datadir, file_name),
            header=False,
            index=False,
            index_label=False,
        )

    def column_name_mapping(self, df):
        assert len(df.columns) == len(
            self.csv_columns
        ), "Input CSV column and expected column amount mismatch"

        return df.rename(columns={i: j for i, j in enumerate(self.csv_columns)})

    def _write_label_error_log(self, df):
        error_df = df[df["label_valid"] != "Valid"].drop_duplicates("file_name")

        if not error_df.empty:
            warning_log = error_df.apply(
                lambda frame: f"{frame['file_name']} can not be used due to {frame['label_valid']} error.",
                axis=1,
            )
            warning_message = "FOLLOWING STUDIES ARE NOT VALID\n"
            warning_message += "\n".join(warning_log.values)

            print(warning_message)

    def run(self, file_map):
        """
        Processes the given csv file, and performs a series of checks and analysis based on the specified 'column_ops'.
        Returns a dictionary of statistics obtained from the csv file.
        """
        # If predict only, generate dummy inference.csv
        if self.predict_only:
            self._generate_file(file_map)
            return None

        # Read label.csv to preprocess
        # specify label name type as string
        df = pd.read_csv(self.label_path, header=None, dtype={1: str})
        df = self.column_name_mapping(df)
        df["original_name"] = df.loc[:, "file_name"]

        df = self._parse_alias_label_name(df)

        if file_map:
            df.loc[:, "file_name"] = df.loc[:, "file_name"].apply(
                lambda name: file_map[name]
            )

        # check file_name in label.csv is available
        df = self.file_validator.validate(df)

        # check annotations in label.csv are valid
        df = self._check_label_valid(df)
        self._write_label_error_log(df)
        df = df.groupby("file_name").filter(
            lambda frame: (frame["label_valid"] == "Valid").all()
        )

        # create label_name.csv for this dataset
        self.class_list, self.class_num = self._get_class_list_from_df(df)

        if self.class_num < self.min_class_num:
            raise Exception(f"Labels require at least {self.min_class_num} classes.")
        
        with open(self.pretrained_model_json, "r") as f:
            model_info = json.load(f)
            model_class_list = model_info["class_list"]
            assert set(self.class_list).issubset(
                set(model_class_list)
            ), f"provided class_list: {self.class_list} and model class_list: {model_class_list} are mismatched"
            self.class_list = model_class_list

        pd.Series(self.class_list).to_csv(
            self.classes_file_path,
            header=False,
            index=False,
            index_label=False,
        )
        self.class_map = {lab: i for i, lab in enumerate(self.class_list)}

        df = self.process_label_info(df)

        self._df_to_file(df, "inference")
        return {"inference": df}

    @abstractmethod
    def _check_label_valid(self, df):
        pass

    @abstractmethod
    def process_label_info(self, df):
        pass

    def _parse_alias_label_name(self, df):
        return df


class MultiLabelPreprocess(Preprocess):
    def __init__(self, args, file_validator):
        super().__init__(args, file_validator)
        self.min_class_num = 1
        self.csv_columns = [
            "file_name",
            "label",
            "binary_label",
        ]
        self.output_csv_columns = [
            "file_name",
            "original_name",
            "file_width",
            "file_height",
            "label",
            "binary_label",
        ]

    def _check_label_valid(self, df):
        unique_class_set = df.groupby("file_name").agg({"label": set})["label"].mode()
        if len(unique_class_set) > 1:
            df["label_valid"] = "The_Class_Set_Is_Not_Well_Defined"
            return df

        def class_set_equal(frame):
            frame["label_valid"] = "Some_Classes_Are_Invalid"

            if sorted(frame["label"].tolist()) == sorted(list(unique_class_set[0])):
                frame["label_valid"] = "Valid"

            return frame

        return df.groupby("file_name").apply(class_set_equal)

    def process_label_info(self, df):
        # keep track of multilabel information
        return df.sort_values(["file_name", "label"])


class SingleLabelPreprocess(Preprocess):
    def __init__(self, args, file_validator):
        super().__init__(args, file_validator)
        self.min_class_num = 2
        self.csv_columns = ["file_name", "label"]
        self.output_csv_columns = [
            "file_name",
            "original_name",
            "file_width",
            "file_height",
            "label",
            "mapped_label",
        ]

    def _check_label_valid(self, df):
        tmp_df = df.set_index("file_name")
        tmp_df["label_valid"] = df.groupby("file_name").apply(
            lambda frame: len(frame) == 1
        )
        tmp_df["label_valid"] = tmp_df["label_valid"].map(
            {True: "Valid", False: "Multiple_Label_For_The_File"}
        )
        return tmp_df.reset_index()

    def process_label_info(self, df):
        # generate dataframe with mapped integer labels
        df["mapped_label"] = df["label"].replace(self.class_map)
        return df


class DetectionPreprocess(Preprocess):
    def __init__(self, args, file_validator):
        super().__init__(args, file_validator)
        self.min_class_num = 0
        self.csv_columns = [
            "file_name",
            "label",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_width",
            "bbox_height",
        ]
        self.output_csv_columns = [
            "file_name",
            "original_name",
            "file_width",
            "file_height",
            "label",
            "bbox",
        ]

    def _check_label_valid(self, df):
        file_height, file_width = df["file_height"], df["file_width"]

        bbox_x_max = df["bbox_x_min"] + df["bbox_width"]
        bbox_y_max = df["bbox_y_min"] + df["bbox_height"]

        bbox_x_valid = (
            (df["bbox_width"] > 0)
            & (0 <= df["bbox_x_min"])
            & (bbox_x_max <= file_width)
        )
        bbox_y_valid = (
            (df["bbox_height"] > 0)
            & (0 <= df["bbox_y_min"])
            & (bbox_y_max <= file_height)
        )
        no_bbox = df["label"] == self.no_annotation_label

        bbox_valid = (bbox_x_valid & bbox_y_valid) | no_bbox

        df["label_valid"] = bbox_valid.map(
            {True: "Valid", False: "BBox_Is_Outside_File_Or_Missing_Some_Values"}
        )

        # handle valid bounding boxs which x2 and y2 are on the edge
        df["bbox_x_min"] = df["bbox_x_min"].clip(0, file_width - 2)
        df["bbox_y_min"] = df["bbox_y_min"].clip(0, file_height - 2)
        df["bbox_width"] = bbox_x_max.clip(0, file_width - 1) - df["bbox_x_min"]
        df["bbox_height"] = bbox_y_max.clip(0, file_height - 1) - df["bbox_y_min"]

        return df

    def _bbox_converter(self, row):
        if row["label"] == self.no_annotation_label:
            return json.dumps({})

        bbox = row.iloc[1:6].to_dict()
        bbox["label"] = self.class_map[row[1]]

        return json.dumps(bbox)

    def process_label_info(self, df):
        # process bounding box information
        df["bbox_area"] = df["bbox_width"] * df["bbox_height"]
        df["bbox_w_div_h"] = df["bbox_width"] / df["bbox_height"]

        df["bbox"] = df.apply(self._bbox_converter, axis=1)

        return df

    def _parse_alias_label_name(self, df):
        alias_name_map = {"negative": NO_ANNOTATION_LABEL}
        df["label"] = df["label"].apply(
            lambda x: alias_name_map[x] if (x in alias_name_map) else x
        )
        return df


class SegmentationPreprocess(Preprocess):
    def __init__(self, args, file_validator):
        super().__init__(args, file_validator)
        self.min_class_num = 1
        self.csv_columns = [
            "file_name",
            "label",
            "mask_width",
            "mask_height",
            "mask",
        ]
        self.output_csv_columns = [
            "file_name",
            "original_name",
            "file_width",
            "file_height",
            "label",
            "mask",
        ]

    def _check_label_valid(self, df):
        unique_class_set = df.groupby("file_name").agg({"label": set})["label"].mode()
        if len(unique_class_set) > 1:
            df["label_valid"] = "The_Class_Set_Is_Not_Well_Defined"
            return df

        def check_mask_valid(frame):
            class_valid = False
            if sorted(frame["label"].tolist()) == sorted(list(unique_class_set[0])):
                class_valid = True

            height_valid = frame["mask_height"] == frame["file_height"]
            width_valid = frame["mask_width"] == frame["file_width"]
            shape_valid = (height_valid & width_valid).all()

            frame["label_valid"] = (
                "Valid" if class_valid and shape_valid else "Some_Masks_Are_Invalid"
            )

            return frame

        return df.groupby("file_name").apply(check_mask_valid)

    def _transform_nan_mask(self, mask, mask_width, mask_height):
        if not isinstance(mask, str) and math.isnan(mask):
            return encode(np.array([[0] * mask_height] * mask_width))

        return mask

    def process_label_info(self, df):
        df = df.sort_values(["file_name", "label"])

        df["mask"] = df.apply(
            lambda row: self._transform_nan_mask(
                row["mask"], row["mask_width"], row["mask_height"]
            ),
            axis=1,
        )
        df["mask_area_percentage"] = df.apply(
            lambda row: mask_area_percentage(
                row["mask"], row["mask_width"], row["mask_height"]
            ),
            axis=1,
        )

        return df
