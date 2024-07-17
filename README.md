# DeepQ AI Platform - Local Inference
## Requirements

1. Please prepare a Ubuntu Focal 20.04 (LTS) machine
2. Please make sure the local nvidia driver version should be >=450.80.02
3. Please make sure you have installed the cuda toolkit 11.7
4. Please make sure you have installed the cudnn 8
5. Please make sure you have a clean python3 environment which version should be >=3.9.13

## Inference
1. Prepare inference source code:
   ```bash
   cd ~
   mkdir inference && cd $_
   ```
   And git clone this repository

2. install packages:
   (Please be careful to not install the packages in your working environment)
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare dataset for inference:
   - create a `dataset` folder under `src` and put your data inside
   ```
   the final dataset folder would be like
   dataset
   ├── image
   │   ├── 1.png
   │   ├── 2.png
   │   └── 3.png
   └── label.csv (optinal)
   ```
   - you could visit the document for more information

4. Prepare trained model
   - download the trained model from AIP and you would get the `your_model_name.zip`
   - create a `pretrain` folder under `src`, put the `your_model_name.zip` inside and unzip it
   ```
   the final pretrain folder would be like
   pretrain
   ├── result.zip
   ├── model.onnx
   └── model.json
   ```

5. Modify the `bath_inference.json`
  - Tags in `main_params`:
    - The `task_type` should be [`single_label`, `multi_label`, `detection`, `segmentation`]
    - The `file_type` should be [`jpg_png`, `dicom`]
    - The `confidence_threshold` should be a float in [0, 1]. The default value for `single_label`, `multi_label` and `segmentation` is 0.5. The default value for `detection` is 0.7 (0.25 for yolo).
  - Tags in `others`:
    - The `metrics` is `true` means calculate the inference score (you have to provide `label.csv` under `dataset`)
    - The `cuda` is `true` means that you want to inference with GPU

6. Run the command `bash batch_inference.sh batch_inference.json`

7. Get the inference result under `src/result` folder
