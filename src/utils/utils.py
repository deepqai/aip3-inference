"""
2018-2019 HTC Corporation. All Rights Reserved.
This source code is licensed under the HTC license which can be found in the
LICENSE file in the root directory of this work.
"""
import argparse
import ast
import os
import secrets
import urllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import simplejson as json
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .factory import get_task_object
from .transform import AIP_Augmentation


def read_config(config_file):
    """Read settings from config file."""
    with open(config_file, "r") as op:
        return json.load(op)


def to_device(data, device=torch.device("cpu"), detach=False):
    """
    Recursively send all tensor to device in different kinds of container types.

    :param data: (Any) the tensor or container include tensor
    :param device: (torch.device) the device to send the Tensor to
    :param detach: (bool) whether to detach the Tensor from the current device
    """

    def to_device_map(data):
        if isinstance(data, torch.Tensor):
            if detach:
                data = data.detach()
            return data.to(device)
        elif isinstance(data, tuple):
            return tuple([to_device_map(x) for x in data])
        elif isinstance(data, list):
            return [to_device_map(x) for x in data]
        elif isinstance(data, dict):
            return {k: to_device_map(v) for k, v in data.items()}
        else:
            return data

    return to_device_map(data)


def track_progress_stats(logger, model_output, **additional_data):
    """tracks stats in the progress using Logger

    Args:
        logger (Logger): the Logger needed to track training progress
        loss (torch.Tensor): the summed loss of the model
        predictions (Any): the model's predictions
    """

    with torch.no_grad():

        iteration_stats = {}

        iteration_stats["predictions"] = model_output["predictions"]
        iteration_stats.update(additional_data)

        logger.update_epoch_info(iteration_stats)


def load_dataset(
    main_params, datadir, label_name, aug_params=None, predict_mode=False
):
    assert os.path.exists(datadir), f"Given data directory: {datadir} is not valid!"

    task = get_task_object(main_params["task_type"])
    Dataset, data_collate_fn, image_size = task.get_dataset_setting(
        main_params["model_name"]
    )

    aip_aug = AIP_Augmentation(
        task_type=main_params["task_type"],
        aug_params=aug_params,
        image_size=image_size,
        predict_mode=predict_mode,
    )

    transform = aip_aug.get_augmentation()

    # doing the preprocessing on data (2d 2.5d etc.)
    record_list = get_input_records(main_params, f"{os.path.join(datadir, label_name)}")

    dataset = Dataset(datadir=datadir, record_list=record_list, transform=transform)

    return DataLoader(
        dataset=dataset,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collate_fn,
    )


def concat_series(df, num_slice=1):
    """Concatenate the series image.

    :param df: (pandas.DataFrame) a DataFrame contain `file_name` columns
    :param num_slice: (int) number of slice

    :return (panda.DataFrame) add the `file_path` column, include {num_slice} image path
    """
    df["study_name"] = df.apply(
        lambda df: str(Path(df["file_name"])).split("/")[-2], axis=1
    )
    df = df.sort_values(["study_name", "file_name"])
    group_df = df.groupby("study_name", as_index=False, sort=False)
    file_path = []

    pad_width = (
        (num_slice // 2, num_slice // 2)
        if num_slice % 2 == 1
        else (num_slice // 2, num_slice // 2 - 1)
    )
    for _, group in group_df:
        path = group["file_name"]
        path = np.pad(path, pad_width=pad_width, mode="edge")
        for idx in range(pad_width[0], len(path) - pad_width[1]):
            start_idx = idx - pad_width[0]
            end_idx = start_idx + num_slice
            file_path.append(list(path[start_idx:end_idx]))
    df["file_path"] = file_path
    df = df.sort_values("original_name")

    return df


def get_first_element(frame):
    return frame.iloc[0]


def get_input_records(main_params, label_path):
    """Pre-process csv for Dataset input.

    :param main_params: parameters of main setting
    :param datadir: (str) data directory path

    :return record_list: list of dictionary records
    """
    file_type = main_params["file_type"]
    task_type = main_params["task_type"]

    csv_columns = [
        "file_name",
        "original_name",
        "file_width",
        "file_height",
        "label_name",
        "label",
    ]

    if not os.path.isfile(label_path):
        raise Exception(label_path + " is not a valid file.")

    df = pd.read_csv(label_path, header=None, encoding="utf-8")

    num_columns = len(df.columns)

    df = df.rename(
        columns={
            idx: column_name for idx, column_name in enumerate(csv_columns[:num_columns])
        }
    )

    # 4 means the number of csv_columns without label information
    if num_columns == 4:
        df["label"] = None
    else:
        df = (
            df.groupby("file_name", sort=False)
            .agg(
                {
                    "original_name": get_first_element,
                    "file_width": get_first_element,
                    "file_height": get_first_element,
                    "label": list,
                }
            )
            .reset_index()
        )

    if file_type == "jpg_png":
        df["file_path"] = df["file_name"].apply(lambda name: [name])
    elif file_type == "dicom":
        df = concat_series(df, 3)
    else:
        raise Exception(file_type + " is not a valid file type")

    record_list = df[
        [
            "file_path",
            "original_name",
            "file_width",
            "file_height",
            "label",
        ]
    ].to_dict(orient="records")

    return record_list


def flatten_directory(dir):
    """Flattens a given folder with AIP v2 (folder structured) into an AIP v3 format (flatten).
    Useful for debugging.

    :param dir: (str) directory to the folder containing images

    :return file_mapper: (dict) a map for files from the original v2 format to the new v3 format
    """

    # collect all the files under this current structure
    file_list = []
    for root, _, files in os.walk(dir, topdown=False):

        for f in files:
            filepath_rel = os.path.relpath(
                os.path.join(root, f), dir
            )  # relative file path
            file_list.append(filepath_rel)

    # copy the files to 'dir', rename the files, create mapper
    file_map = {}
    for file in file_list:

        # reformate the names of the files
        reformat_name = urllib.parse.quote(file, safe="")  # file.replace("/", "%2F")

        # create file mapper
        file_map[file] = reformat_name

        # copy and rename files to the directory
        os.rename(os.path.join(dir, file), os.path.join(dir, reformat_name))

    # remove the empty directories
    for root, _, files in os.walk(dir, topdown=False):

        if not os.listdir(root):
            os.rmdir(root)

    return file_map


def dicom_to_pil(dcm_file, correct_monochrome=True):
    """Extracts dicom file's pixel array, apply windowing, and return PIL Image"""

    sitk_img = reader.Execute()
    # the coordinate of pixel array would be (z, y, x)
    pixel_array = sitk.GetArrayFromImage(sitk_img)[0]

    if "0028|1050" in sitk_img and "0028|1051" in sitk_img:
        window_center = float(sitk_img["0028|1050"].strip().split("\\")[0])
        window_width = float(sitk_img["0028|1051"].strip().split("\\")[0])
        pixel_array = pixel_array.clip(
            window_center - window_width // 2, window_center + window_width // 2
        )

    pixel_array = np.uint8(
        (pixel_array - pixel_array.min())
        / (pixel_array.max() - pixel_array.min())
        * 255
    )

    if correct_monochrome and sitk_img["0028|0004"].strip() == "MONOCHROME1":
        pixel_array = 255 - pixel_array  # invert color

    return Image.fromarray(pixel_array).convert("RGB")


def generate_uid(prefix="1.2.826.0.1.3680043.8.498."):
    maximum = 10 ** (64 - len(prefix))
    # randbelow is in [0, maximum)
    return f"{prefix}{secrets.randbelow(maximum)}"[:64]


def read_dcm_info(dcm_file):
    """Extracts the series_id and instance number of the dicom file"""
    metadata_keys = reader.GetMetaDataKeys()

    # if no study or series, generate random uid for them
    study_id, series_id = generate_uid(), generate_uid()
    if "0020|000d" in metadata_keys:
        study_id = reader.GetMetaData("0020|000d").strip()
    if "0020|000e" in metadata_keys:
        series_id = reader.GetMetaData("0020|000e").strip()

    if "0020|0013" in metadata_keys:
        instance = int(reader.GetMetaData("0020|0013").strip())
    else:
        # else we treat as individual 2d slices
        instance = 0

        if "0008|0018" in metadata_keys:
            # if SOP UID is available, append to series_id
            sop_uid = reader.GetMetaData("0008|0018").strip()
            series_id = f"{series_id}--{sop_uid}"

    # get view position
    view_position = ""
    if "0018|5101" in metadata_keys:
        view_position = reader.GetMetaData("0018|5101").strip()

    # read dicom pixel array's height and width
    width, height, _ = reader.GetSize()

    return study_id, series_id, instance, view_position, height, width


def restructure_dicom_folder(img_folder, convert_to_png=True):
    """Restructures a given folder containing images (all flattened at root directory)
    into folders split according to the series tag in each of the dicom files.
    Optionally converts each of the dicom files into png files.

    Returns a dictionary that maps each file from the original flattened format into
    folder structured format.

    :param img_folder: (str) directory to the folder containing images (assumed to be flattened)
    :param conver_to_png: (bool) default=True, converts the dicom to png images

    :return file_mapper: (dict) maps each file from the original flattened format into
    folder structured format
    """

    _all_files = os.listdir(img_folder)  # get a list of all files (all flattened)
    file_map = {}  # keep track of which files are converted to which files
    reader = sitk.ImageFileReader()

    for file in _all_files:

        reader.SetFileName(os.path.join(img_folder, file))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        study_id, series_id, instance, view_position, height, width = read_dcm_info(
            reader
        )

        save_folder = os.path.join(
            img_folder, f"{study_id}-{series_id}-{view_position}-{height}-{width}"
        )  # the folder to save this file in
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(
            save_folder,
            f"{study_id}-{series_id}-{view_position}-{height}-{width}_{instance:04}",
        )

        if convert_to_png:
            dcm_img = dicom_to_pil(reader, correct_monochrome=True)
            save_path = save_path + ".png"
            dcm_img.save(save_path)  # write converted png to disk
            os.remove(
                os.path.join(img_folder, file)
            )  # remove original file to save space
        else:
            save_path = save_path + ".dcm"
            os.rename(os.path.join(img_folder, file), save_path)

        file_map[file] = os.path.relpath(save_path, img_folder)  # add to file_map

    # save a copy of the filemap to disk for backup
    with open(os.path.join(img_folder, "restructure_log.json"), "w") as f:
        f.write(json.dumps(file_map, indent=4))

    return file_map


def get_result_statistics_in_metrics(metrics):
    result = defaultdict(list)
    for metric_name, metric in metrics.items():
        if metric_name == "average":
            continue

        if metric_name == "confusion_matrix":
            result[metric_name] = metric["matrix"]
        elif metric_name == "summary":
            new_metric = []
            for sub_metric_name, sub_metric in metric.items():
                sub_metric.update({"name": sub_metric_name})
                new_metric.append(sub_metric)
            result[metric_name] = new_metric
        else:
            metric.update({"name": metric_name})
            metric.pop("loss", None)
            result["labels"].append(metric)

    return result
