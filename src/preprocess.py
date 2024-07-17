import argparse
import fcntl
import logging
import os
import random
import time
from pathlib import Path

from constants.default import DEFAULT_DATA_DIR
from constants.preprocess import IMG_FOLDER, LABEL_FILE
from utils.factory import get_file_validator, get_task_object
from utils.utils import read_config, restructure_dicom_folder


class Flocker:
    """File locker, avoid race condition problem."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.fd_file = None

    def __enter__(self):
        self.fd_file = os.open(self.file_name, os.O_CREAT, 0o664)
        fcntl.flock(self.fd_file, fcntl.LOCK_EX)
        return self.fd_file

    def __exit__(self, *args):
        fcntl.flock(self.fd_file, fcntl.LOCK_UN)
        os.close(self.fd_file)
        self.fd_file = None


def preprocess(args):
    """
    Pre-processes the data in args.datadir.

    args: (argparse) arguments passed from argument parser
    """
    random.seed(args.random_seed)

    # check datadir valid
    img_folder_dir = os.path.join(args.datadir, IMG_FOLDER)
    assert os.path.exists(
        img_folder_dir
    ), f"'image' folder not found in the given directory: {img_folder_dir}"

    # check label.csv exists for train mode
    label_path = os.path.join(args.datadir, LABEL_FILE)
    if not args.predict:
        assert os.path.isfile(
            label_path
        ), f"Label file: {LABEL_FILE} not found in given directory: {args.datadir}"

    # perform file locking
    with Flocker(os.path.join(args.datadir, "flock.lock")):
        if not args.force and os.path.isfile(
            os.path.join(args.datadir, f"_inference_processed.flag")
        ):
            print(f"{args.datadir} had already been processed for inference.")
            return

        # print preprocess arguments
        print("Preprocess Args:")
        for arg_param in vars(args):
            print(f"- {arg_param}: {vars(args)[arg_param]}")

        start_time = time.time()  # time preprocess

        file_map = None
        if args.file_type == "dicom":
            file_map = restructure_dicom_folder(img_folder_dir)

        file_validator = get_file_validator(args.file_type, args.datadir)
        task = get_task_object(args.task_type)

        preprocess = task.get_preprocess(args, file_validator)

        # process csv file
        process_log = preprocess.run(file_map)

        if process_log:
            data_insight = task.get_data_insight(args.datadir, preprocess.class_map)
            data_statistics = {}
            for split_name, stats in process_log.items():
                data_statistics[split_name] = data_insight.run(split_name, stats)

        print(f"Preprocessed completed in {time.time() - start_time} seconds.")

        Path(os.path.join(args.datadir, f"_inference_processed.flag")).touch()

        if process_log:
            return {"data_statistics": data_statistics, "dataset_info": process_log}


def get_arguments(seed):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Pass --predict option to not preprocess label.csv",
    )
    args = parser.parse_args()

    args.force = (
        False  # if True, ignores existing preprocess, and re-runs preprocessing again
    )
    args.random_seed = seed  # random seed to use

    config = read_config(args.config)
    if "others" in config.keys():
        args.inference = True
        args.metrics = config["others"]["metrics"]
    args.file_type = config["main_params"]["file_type"]
    args.task_type = config["main_params"]["task_type"]
    args.val_ratio = config["main_params"].get("val_ratio", 0)
    args.datadir = DEFAULT_DATA_DIR

    return args


if __name__ == "__main__":
    seed = 9527
    args = get_arguments(seed)

    preprocess(args)
