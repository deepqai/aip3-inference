import torch

from utils import utils as U


class ModelController:
    def prepare_output(model_outputs):
        """takes the raw output of the object detection models and formats it for remaning procedures

        Returns:
            loss: (torch.Tensor) the summed loss of the model
            predictions: (list) a list of predictions from the model
        """

        losses, predictions = model_outputs  # known beforehand, tailored for detection

        output = {}

        if losses:
            # prepare loss
            loss_weights = {k: 1 for k in losses.keys()}
            output["loss"] = sum(
                [losses[k] * loss_weights[k] for k in losses.keys()]
            )  # total loss

        output["predictions"] = [predictions]

        return output

    def extract_additional_data(batch_data, heatmap_utils=None):
        """extracts additional data from batch_data that doesn't go through model

        Args:
            targets (torch.Tensor): the targets for each image in the batch
            filenames (list): a list of the filenames
        """

        additional_data = {}

        additional_data["file_paths"] = [batch_data["file_paths"]]
        additional_data["file_name"] = [batch_data["filenames"]]
        additional_data["num_examples"] = [len(batch_data["images"])]

        if "targets" in batch_data:
            additional_data["targets"] = [batch_data["targets"]]

        return additional_data


class DetectionModelController(ModelController):
    def prepare_output(model_outputs):
        """takes the raw output of the object detection models and formats it for remaning procedures

        Returns:
            loss: (torch.Tensor) the summed loss of the model
            predictions: (list) a list of predictions from the model
        """

        losses, predictions = model_outputs  # known beforehand, tailored for detection

        output = {}

        if losses:
            # prepare loss
            loss_weights = {k: 1 for k in losses.keys()}
            output["loss"] = sum(
                [losses[k] * loss_weights[k] for k in losses.keys()]
            )  # total loss

        # prepare predictions
        output["predictions"] = U.to_device(
            predictions, torch.device("cpu"), detach=True
        )

        return output

    def extract_additional_data(batch_data, heatmap_utils=None):
        """extracts additional data from batch_data that doesn't go through model

        Args:
            targets_list (list): a list of the targets for each image in the batch
            filenames (list): a list of the filenames
        """

        additional_data = {}

        additional_data["file_paths"] = batch_data["file_paths"]
        additional_data["file_name"] = batch_data["filenames"]
        additional_data["num_examples"] = [len(batch_data["images"])]

        if "targets" in batch_data:
            additional_data["targets"] = U.to_device(
                batch_data["targets"], torch.device("cpu"), detach=True
            )

        return additional_data


class SegmentationModelController(ModelController):
    def prepare_output(model_outputs):
        """takes the raw output of the object detection models and formats it for remaning procedures

        Returns:
            loss: (torch.Tensor) the summed loss of the model
            predictions: (list) a list of predictions from the model
        """
        losses, predictions = model_outputs

        output = {}

        if losses:
            output["loss"] = losses["all_loss"]

        output["predictions"] = predictions

        return output

    def extract_additional_data(batch_data, heatmap_utils=None):
        """extracts additional data from batch_data that doesn't go through model

        Args:
            targets (torch.Tensor): the targets for each image in the batch
            filenames (list): a list of the filenames
        """

        additional_data = {}

        additional_data["file_paths"] = [batch_data["file_paths"]]
        additional_data["file_name"] = [batch_data["filenames"]]
        additional_data["num_examples"] = [len(batch_data["images"])]
        additional_data["width"] = [batch_data["width"].numpy().tolist()]
        additional_data["height"] = [batch_data["height"].numpy().tolist()]

        if "targets" in batch_data:
            targets = batch_data["targets"]
            additional_data["targets"] = torch.stack(targets, dim=1)

        return additional_data
