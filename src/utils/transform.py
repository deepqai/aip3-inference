import albumentations as A
from albumentations.pytorch import ToTensorV2


class AIP_Augmentation:
    def __init__(self, task_type, aug_params, image_size, predict_mode=False):
        self.task_type = task_type
        self.aug_params = aug_params
        self.image_size = image_size
        self.predict_mode = predict_mode

        self.default_augmentation = [self._resize(), self._totensor()]
        if self.task_type != "detection":
            self.default_augmentation.insert(1, self._normalize())

        self.additional_params = dict()
        if self.task_type == "detection" and not self.predict_mode:
            self.additional_params["bbox_params"] = A.BboxParams(
                format="coco", label_fields=["labels"]
            )

    def get_augmentation(self):
        return A.Compose(self.default_augmentation, **self.additional_params)

    def _normalize(self):
        return A.Normalize()

    def _totensor(self):
        return ToTensorV2()

    def _resize(self):
        return A.Resize(height=self.image_size, width=self.image_size)
