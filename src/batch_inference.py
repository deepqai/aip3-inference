import argparse
import copy
import json
import os
from argparse import Namespace

import onnx
import onnxruntime
import pandas as pd
import torch

import utils.utils as U
from constants.metric import REALTIME_STATISTICS_NAME_MAPPING
from constants.default import DEFAULT_DATA_DIR, DEFAULT_PRETRAIN_DIR, DEFAULT_RESULT_DIR
from data_insights.generate_pdf import InsightReport
from utils.factory import get_file_validator, get_task_object
from utils.utils import get_result_statistics_in_metrics, read_config


def torch_to_onnx_numpy(tensor):
    return tensor.detach().numpy() if tensor.requires_grad else tensor.numpy()


def batch_inference(args):
    config = {}

    config["main_params"] = args.config["main_params"]
    config["others"] = args.config["others"]

    with open(f"{DEFAULT_PRETRAIN_DIR}/model.json", "r") as f:
        model_info = json.load(f)

    config["main_params"]["model_name"] = model_info["architecture_name"]

    class_list = model_info["class_list"]
    n_classes = len(class_list)
    class_map = {class_name: idx for idx, class_name in enumerate(class_list)}

    # load logger and model controller
    task_type = config["main_params"]["task_type"]
    task = get_task_object(task_type)
    inference_logger = task.get_logger(
        DEFAULT_RESULT_DIR,
        class_list,
        calculate_metrics=config["others"]["metrics"],
        confidence_threshold=config["main_params"]["confidence_threshold"],
    )

    model_controller_class = task.get_controller()
    model_controller = model_controller_class()

    model_path = f"{DEFAULT_PRETRAIN_DIR}/model.onnx"
    try:
        onnx.checker.check_model(onnx.load(model_path))
    except onnx.checker.ValidationError as e:
        raise ValueError(f"ONNX model is not valid: {e}")

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}))

    ort_session = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
    )

    batch_inference_dataloader = U.load_dataset(
        main_params=config["main_params"],
        datadir=DEFAULT_DATA_DIR,
        label_name="inference.csv",
        predict_mode=not config["others"]["metrics"],
    )

    inference_logger.clean_epoch_info()

    for i, batch_data in enumerate(batch_inference_dataloader):
        with torch.no_grad():
            # move to device
            batch_data = copy.deepcopy(batch_data)

            input_images = batch_data["images"]
            if task_type == "detection":
                input_images = torch.stack(input_images)

            ort_inputs = {ort_session.get_inputs()[0].name: torch_to_onnx_numpy(input_images)}
            ort_outputs = ort_session.run(None, ort_inputs)

            outputs = torch.Tensor(ort_outputs[0])
            if task_type == "detection":
                outputs = [{"boxes": torch.Tensor(ort_outputs[0]), "labels": torch.Tensor(ort_outputs[1]).int(), "scores": torch.Tensor(ort_outputs[2])}]
            model_outputs = ({}, outputs)

            output_data = model_controller_class.prepare_output(
                model_outputs
            )  # prepares output data

            additional_data = model_controller_class.extract_additional_data(
                batch_data,
            )
            U.track_progress_stats(inference_logger, output_data, **additional_data)

            print({"status": "running", "progress": int((i + 1) / len(batch_inference_dataloader) * 100)})

    output = {"status": "finishing", "progress": 100}

    metrics = case_study_examples = None
    if config["others"]["metrics"]:
        target_metric = REALTIME_STATISTICS_NAME_MAPPING[task_type]["score"]
        metrics = inference_logger.get_metrics()
        case_study_examples = inference_logger.get_case_study_examples()

        output["score"] = metrics["average"][target_metric]

    prediction_df = inference_logger.prediction_to_dataframe()

    preprocess_argument = config["main_params"]
    preprocess_argument["inference"] = True
    preprocess_argument["val_ratio"] = 0
    preprocess_argument["datadir"] = DEFAULT_DATA_DIR
    file_validator = get_file_validator(
        config["main_params"]["file_type"], DEFAULT_DATA_DIR
    )

    preprocess = task.get_preprocess(Namespace(**(preprocess_argument)), file_validator)
    preprocess.class_map = class_map

    if task_type == "multi_label":
        pred_stats = prediction_df.copy()
    elif task_type == "segmentation":
        pred_stats = prediction_df.iloc[:, :-2].copy()
    else:
        pred_stats = prediction_df.iloc[:, :-1].copy()
    pred_stats.columns = range(pred_stats.columns.size)
    pred_stats = preprocess.column_name_mapping(pred_stats)
    pred_stats = preprocess.process_label_info(pred_stats)

    data_insight = task.get_data_insight(DEFAULT_DATA_DIR, class_map)
    data_insight.run("prediction", pred_stats)

    # clear buffers
    inference_logger.clean_epoch_info()

    os.makedirs(DEFAULT_RESULT_DIR, exist_ok=True)

    prediction_df.to_csv(f"{DEFAULT_RESULT_DIR}/prediction.csv", index=None)

    if metrics is not None:
        inference_statistics = {"inference": get_result_statistics_in_metrics(metrics)}

        report_path = f"{DEFAULT_RESULT_DIR}/inference_insight_report.pdf"
        insight_report = InsightReport(
            filename=report_path,
            task_type=task_type,
            data_insight_dir=f"{DEFAULT_DATA_DIR}/data_insight",
            insight_content_json="inference_insight_content.json",
            report_setting_json="report_setting.json",
            datadir=DEFAULT_DATA_DIR,
            class_list=class_list,
            inference_statistics=inference_statistics,
            case_study_examples=case_study_examples,
        )
        insight_report.run()
    
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)

    args = parser.parse_args()
    args.config = read_config(args.config)

    batch_inference(args)
