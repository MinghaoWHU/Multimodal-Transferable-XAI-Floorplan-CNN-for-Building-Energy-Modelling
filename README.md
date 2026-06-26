# Multimodal Transferable XAI Floorplan-CNN for Building Energy Modelling

A clean and reusable PyTorch implementation of **SimpleCNNGAP9** for floor-plan-based building energy prediction. The project reorganizes the original research scripts into a structured deep learning workflow, including model definition, dataset loading, training, SHAP-based model explanation, and convolutional response map visualization.

The model is designed for early-stage building energy analysis by jointly using floor-plan images and tabular energy-efficiency features.

---

## 1. Project Overview

This project contains four main components:

1. **Model architecture**  
   `SimpleCNNGAP9` combines multi-scale convolutional feature extraction, global average pooling, an image-derived scalar feature, and tabular feature fusion.

2. **Dataset loading**  
   The data pipeline supports `train1.csv`, `test1.csv`, and an optional `merged_data.csv` for full-sample explanation and visualization.

3. **Model training**  
   The training script supports automatic mixed precision, checkpoint resuming, learning-rate scheduling, best-model saving, training-log export, rollback, and optional zero-image augmentation.

4. **Model interpretation**  
   The project supports SHAP feature contribution analysis and convolutional response map visualization to explain how floor-plan geometry and tabular features influence the prediction.

---

## 2. Public Dataset and Pretrained Weights

The floor-plan energy dataset for **London and Birmingham** is available on Kaggle:

```text
https://www.kaggle.com/datasets/minghao66666/floorplan-birmingham-and-london
```

The pretrained `SimpleCNNGAP9` checkpoint, together with comparison checkpoints for ViT and EfficientNet, is available on Kaggle:

```text
https://www.kaggle.com/datasets/minghao66666/simplecnngap9-checkpoint-vs-vit-and-efficientnet
```

After downloading the dataset and checkpoint files, place them in the project directory following the recommended structure below.

---

## 3. Project Structure

```text
simplecnn_gap9_project/
│
├── README.md
├── requirements.txt
├── train.py
├── explain_shap.py
├── visualize_conv_maps.py
│
├── src/
│   └── simplecnn_gap9/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       └── utils.py
│
├── checkpoints/
└── outputs/
```

Recommended dataset directory:

```text
plan_1125_dataset/
├── train1.csv
├── test1.csv
├── merged_data.csv
└── image files...
```

Recommended checkpoint directory:

```text
checkpoints/
└── plan_zichuang_gap9/
    ├── best.pth
    ├── last.pth
    └── logs.jsonl
```

---

## 4. Data Format

Each CSV file should follow the structure below:

```text
image_path, feature_1, feature_2, ..., feature_9, target
```

The first column stores the floor-plan image path. The following nine columns store tabular energy-efficiency variables. The last column stores the regression target.

The project follows the original feature setting and uses the following nine tabular variables before feature filtering:

```text
NUMBER_HEATED_ROOMS
CURRENT_ENERGY_EFFICIENCY
HOT_WATER_ENERGY_EFF
ROOF_ENERGY_EFF
WALLS_ENERGY_EFF
WINDOWS_ENERGY_EFF
LIGHTING_ENERGY_EFF
FLOOR_HEIGHT
MAINHEAT_ENERGY_EFF
```

Following the original training logic, the second tabular variable, `CURRENT_ENERGY_EFFICIENCY`, is removed before model input. Therefore, the final model input contains one image-derived convolutional feature and eight tabular features:

```text
Conv
NUMBER_HEATED_ROOMS
HOT_WATER_ENERGY_EFF
ROOF_ENERGY_EFF
WALLS_ENERGY_EFF
WINDOWS_ENERGY_EFF
LIGHTING_ENERGY_EFF
FLOOR_HEIGHT
MAINHEAT_ENERGY_EFF
```

This same feature-selection rule is used consistently in training, SHAP explanation, and convolutional response visualization.

---

## 5. Environment Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python >= 3.9
PyTorch >= 2.0
CUDA optional
```

If SHAP is not installed, run:

```bash
pip install shap
```

---

## 6. Training

A minimal training command is:

```bash
python train.py --data-root plan_1125_dataset --run-name plan_zichuang_gap9
```

A typical training command is:

```bash
python train.py \
  --data-root plan_1125_dataset \
  --run-name plan_zichuang_gap9 \
  --epochs 1000 \
  --batch-size 128 \
  --workers 8 \
  --lr 1e-2 \
  --optimizer adamw \
  --zero-image-prob 0.1
```

Training results are saved to:

```text
checkpoints/plan_zichuang_gap9/
├── best.pth
├── last.pth
└── logs.jsonl
```

To resume training from a saved checkpoint:

```bash
python train.py \
  --data-root plan_1125_dataset \
  --run-name plan_zichuang_gap9 \
  --resume checkpoints/plan_zichuang_gap9/best.pth
```

---

## 7. SHAP Explanation

To explain the full dataset using `merged_data.csv`:

```bash
python explain_shap.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --split all \
  --output-dir outputs/shap_gap9
```

To explain only the test set:

```bash
python explain_shap.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --split test \
  --output-dir outputs/shap_gap9_test
```

The SHAP output directory contains:

```text
outputs/shap_gap9/
├── shap_summary.png
├── shap_values.csv
├── shap_vs_pred.png
└── dependence_plots/
```

The `shap_values.csv` file contains the following fields:

```text
img_path
pred
target
SHAP_Conv
SHAP_NUMBER_HEATED_ROOMS
SHAP_HOT_WATER_ENERGY_EFF
...
expected_value
reconstructed_pred
reconstruction_error
```

`Conv` represents the image-derived convolutional feature. The remaining fields represent the eight tabular features used by the model.

---

## 8. Convolutional Response Map Visualization

To generate multi-scale convolutional response maps for images listed in `merged_data.csv`:

```bash
python visualize_conv_maps.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --csv merged_data.csv \
  --output-dir outputs/conv_maps \
  --every 8
```

To visualize only images whose paths contain a specific keyword:

```bash
python visualize_conv_maps.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --csv merged_data.csv \
  --output-dir outputs/conv_maps \
  --contains "200029898_Flat 5"
```

The convolutional response maps are extracted through `SimpleCNNGAP9.extract_conv_maps()`, ensuring that the visualization uses the same model structure as training and inference.

---

## 9. Main Improvements over the Original Scripts

This project improves the original scripts in the following ways:

1. Removes unused model variants, including `SimpleCNNGAP7`, `SimpleCNNGAP8`, and `SimpleCNNGAP10`.
2. Removes unused dataset and loss definitions, including `CVRMSELoss`, `RegressionTaskData`, `ImageTabularTaskRateData`, and `ImageTabularTaskDataEP`.
3. Removes repeated imports, unused variables, and redundant logic.
4. Separates model definition, data loading, training, explanation, and visualization into independent modules.
5. Fixes the division-by-zero issue when `torch.cuda.device_count() == 0` on CPU-only machines.
6. Standardizes the `SimpleCNNGAP9.forward()` input interface so that training, inference, and explanation remain consistent.
7. Applies the same tabular feature filtering rule across training and SHAP explanation.
8. Names the image-derived feature as `Conv` in SHAP analysis and explains it together with the eight tabular features.
9. Integrates convolutional response map extraction into `SimpleCNNGAP9.extract_conv_maps()` to avoid inconsistent model definitions.
10. Standardizes checkpoint paths, output paths, training logs, and interpretation results.

---

## 10. Troubleshooting

If `shap` is missing, install it with:

```bash
pip install shap
```

If `num_workers` causes an error on Windows, set it to `0`:

```bash
python train.py --workers 0
```

If GPU memory is insufficient, reduce the batch size:

```bash
python train.py --batch-size 32
```

If no CUDA device is available, the project automatically falls back to CPU training and inference.

---

## 11. Suggested Citation and Acknowledgement

If you use this project, please cite or acknowledge the public Kaggle dataset and checkpoint resources:

```text
Floor-plan energy dataset for London and Birmingham:
https://www.kaggle.com/datasets/minghao66666/floorplan-birmingham-and-london

SimpleCNNGAP9 checkpoint and comparison model weights:
https://www.kaggle.com/datasets/minghao66666/simplecnngap9-checkpoint-vs-vit-and-efficientnet
```

---

## 12. License

Please follow the license terms of the original dataset, pretrained checkpoints, and any third-party dependencies used in this project.
