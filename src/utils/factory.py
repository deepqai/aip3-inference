from abc import ABC, abstractmethod

import torch.optim as optim

from .file_validator import ImageValidator


class TaskFactory(ABC):
    @abstractmethod
    def get_preprocess(self, args, file_validator):
        pass

    @abstractmethod
    def get_data_insight(self, datadir, class_map):
        pass

    @abstractmethod
    def get_logger(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        pass

    @abstractmethod
    def get_dataset_setting(self, model_name):
        pass

    def get_controller(self):
        from networks import ModelController

        return ModelController


class SingleLabelFactory(TaskFactory):
    def get_preprocess(self, args, file_validator):
        from .preprocessor import SingleLabelPreprocess

        return SingleLabelPreprocess(args, file_validator)

    def get_data_insight(self, datadir, class_map):
        from data_insights import SingleLabelDataInsight
        from data_insights.utils import SingleLabelInsightExtractor

        return SingleLabelDataInsight(datadir, class_map, SingleLabelInsightExtractor)

    def get_logger(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        from .logger import SingleLabelLogger

        return SingleLabelLogger(
            log_dir, class_list, calculate_metrics, confidence_threshold=0.5,
        )

    def get_dataset_setting(self, model_name):
        from datasets import SingleLabelDataset as Dataset

        data_collate_fn = None
        image_size = 224
        if "incept" in model_name:
            image_size = 299
        return Dataset, data_collate_fn, image_size


class MultiLabelFactory(TaskFactory):
    def get_preprocess(self, args, file_validator):
        from .preprocessor import MultiLabelPreprocess

        return MultiLabelPreprocess(args, file_validator)

    def get_data_insight(self, datadir, class_map):
        from data_insights import MultiLabelDataInsight
        from data_insights.utils import MultiLabelInsightExtractor

        return MultiLabelDataInsight(datadir, class_map, MultiLabelInsightExtractor)

    def get_logger(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        from .logger import MultiLabelLogger

        return MultiLabelLogger(
            log_dir, class_list, calculate_metrics, confidence_threshold=0.5
        )

    def get_dataset_setting(self, model_name):
        from datasets import MultiLabelDataset as Dataset

        data_collate_fn = None
        image_size = 224
        if "incept" in model_name:
            image_size = 299
        return Dataset, data_collate_fn, image_size


class DetectionFactory(TaskFactory):
    def get_preprocess(self, args, file_validator):
        from .preprocessor import DetectionPreprocess

        return DetectionPreprocess(args, file_validator)

    def get_data_insight(self, datadir, class_map):
        from data_insights import DetectionDataInsight
        from data_insights.utils import DetectionInsightExtractor

        return DetectionDataInsight(datadir, class_map, DetectionInsightExtractor)

    def get_logger(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.7,
    ):
        from .logger import DetectionLogger

        return DetectionLogger(
            log_dir, class_list, calculate_metrics, confidence_threshold=0.5
        )

    def get_controller(self):
        from networks import DetectionModelController

        return DetectionModelController

    def get_dataset_setting(self, model_name):
        from datasets import DetectionDataset as Dataset
        from datasets import od_collate_fn as data_collate_fn

        image_size = 512
        return Dataset, data_collate_fn, image_size


class SegmentationFactory(TaskFactory):
    def get_preprocess(self, args, file_validator):
        from .preprocessor import SegmentationPreprocess

        return SegmentationPreprocess(args, file_validator)

    def get_data_insight(self, datadir, class_map):
        from data_insights import SegmentationDataInsight
        from data_insights.utils import SegmentationInsightExtractor

        return SegmentationDataInsight(
            datadir, class_map, SegmentationInsightExtractor
        )

    def get_logger(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        from .logger import SegmentationLogger

        return SegmentationLogger(
            log_dir,
            class_list,
            calculate_metrics,
        )

    def get_controller(self):
        from networks import SegmentationModelController

        return SegmentationModelController

    def get_dataset_setting(self, model_name):
        from datasets import SegmentationDataset as Dataset

        data_collate_fn = None
        image_size = 512
        return Dataset, data_collate_fn, image_size


def get_file_validator(file_type, datadir):
    file_validators = {"jpg_png": ImageValidator, "dicom": ImageValidator}
    # temporarily let dicom use ImageValidator for now

    if file_type not in file_validators:
        raise ValueError(f"No Support File Type: {file_type}")
    return file_validators[file_type](file_type, datadir)


def get_task_object(task_type):
    task_converter = {
        "single_label": SingleLabelFactory(),
        "multi_label": MultiLabelFactory(),
        "detection": DetectionFactory(),
        "segmentation": SegmentationFactory(),
    }

    if task_type not in task_converter:
        raise ValueError(f"No Support Task Type: {task_type}")
    return task_converter[task_type]
