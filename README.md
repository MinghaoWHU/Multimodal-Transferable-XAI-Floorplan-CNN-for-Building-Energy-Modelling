# SimpleCNNGAP9 Project

本项目将原始脚本中与 `SimpleCNNGAP9` 相关的内容重新整理为一个清晰、可复用的 PyTorch 工程。项目包含四个核心部分：

1. **模型架构**：`SimpleCNNGAP9` 多尺度卷积 + GAP + 图像标量特征 + 表格特征融合。
2. **数据集读取**：读取 `train1.csv`、`test1.csv` 和可选的 `merged_data.csv`。
3. **模型训练**：支持 AMP、断点续训、学习率调度、最佳模型保存、训练日志保存、rollback。
4. **模型解释**：支持 SHAP 特征贡献分析和卷积响应图可视化。

---

## 1. 项目结构

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

---

## 2. 数据格式

默认数据目录为：

```text
plan_1125_dataset/
```

该目录下建议包含：

```text
plan_1125_dataset/
├── train1.csv
├── test1.csv
├── merged_data.csv   # 可选，用于全样本解释
└── image files...
```

CSV 的格式要求如下：

```text
image_path, feature_1, feature_2, ..., feature_9, target
```

其中第 1 列为图像路径。后面 9 列为表格变量。最后 1 列为回归目标值。

当前项目沿用原始脚本中的 9 个表格变量标准化参数：

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

训练时沿用原始脚本逻辑，删除第 2 个表格变量 `CURRENT_ENERGY_EFFICIENCY`，因此模型实际输入为：

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

---

## 3. 环境安装

```bash
pip install -r requirements.txt
```

推荐环境：

```text
Python >= 3.9
PyTorch >= 2.0
CUDA 可选
```

---

## 4. 训练模型

最简运行：

```bash
python train.py --data-root plan_1125_dataset --run-name plan_zichuang_gap9
```

常用参数：

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

训练结果会保存到：

```text
checkpoints/plan_zichuang_gap9/
├── best.pth
├── last.pth
└── logs.jsonl
```

断点续训：

```bash
python train.py \
  --data-root plan_1125_dataset \
  --run-name plan_zichuang_gap9 \
  --resume checkpoints/plan_zichuang_gap9/best.pth
```

---

## 5. SHAP 解释

使用 `merged_data.csv` 解释全样本：

```bash
python explain_shap.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --split all \
  --output-dir outputs/shap_gap9
```

使用测试集解释：

```bash
python explain_shap.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --split test \
  --output-dir outputs/shap_gap9_test
```

输出内容：

```text
outputs/shap_gap9/
├── shap_summary.png
├── shap_values.csv
├── shap_vs_pred.png
└── dependence_plots/
```

`shap_values.csv` 包含：

```text
img_path
pred
target
SHAP_Conv
SHAP_NUMBER_HEATED_ROOMS
...
expected_value
reconstructed_pred
reconstruction_error
```

---

## 6. 卷积响应图可视化

对 `merged_data.csv` 中图像生成多尺度卷积响应图：

```bash
python visualize_conv_maps.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --csv merged_data.csv \
  --output-dir outputs/conv_maps \
  --every 8
```

只处理文件路径中包含特定关键词的图像：

```bash
python visualize_conv_maps.py \
  --data-root plan_1125_dataset \
  --checkpoint checkpoints/plan_zichuang_gap9/best.pth \
  --csv merged_data.csv \
  --output-dir outputs/conv_maps \
  --contains "200029898_Flat 5"
```

---

## 7. 主要优化内容

本项目相对于原始脚本做了以下整理：

1. 删除未使用的 `SimpleCNNGAP7`、`SimpleCNNGAP8`、`SimpleCNNGAP10`。
2. 删除未使用的 `CVRMSELoss`、`RegressionTaskData`、`ImageTabularTaskRateData`、`ImageTabularTaskDataEP`。
3. 删除重复导入和无效变量。
4. 将模型、数据、训练、解释拆分为独立模块。
5. 修复无 GPU 时 `torch.cuda.device_count() == 0` 导致除零的问题。
6. 修正 `SimpleCNNGAP9.forward()` 的输入接口，使训练和解释保持一致。
7. 统一表格变量筛选逻辑，避免训练和 SHAP 解释使用不同特征。
8. 将 SHAP 中的图像特征统一命名为 `Conv`，并与 8 个表格变量共同解释。
9. 将卷积响应图提取整合进 `SimpleCNNGAP9.extract_conv_maps()`，避免另写不一致的模型结构。
10. 训练日志、模型保存、可解释性输出路径全部标准化。

---

## 8. 注意事项

如果训练时报错 `No module named shap`，请安装：

```bash
pip install shap
```

如果 Windows 中 `num_workers` 报错，可以先设置为 0：

```bash
python train.py --workers 0
```

如果显存不足，可以降低 batch size：

```bash
python train.py --batch-size 32
```

