import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. 路径设置
# ============================================================

STAT_CSV = r"all_plan_information.csv"
TRAIN_CSV = r"plan_1125_dataset/train1.csv"
TEST_CSV = r"plan_1125_dataset/test1.csv"

OUT_DIR = r"model_compare_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. 字典映射
# ============================================================

age_dict = {
    "England and Wales: before 1900": 1,
    "England and Wales: 1900-1929": 2,
    "England and Wales: 1930-1949": 3,
    "England and Wales: 1950-1966": 4,
    "England and Wales: 1967-1975": 5,
    "England and Wales: 1976-1982": 6,
    "England and Wales: 1983-1990": 7,
    "England and Wales: 1991-1995": 8,
    "England and Wales: 1996-2002": 9,
    "England and Wales: 2003-2006": 10,
    "England and Wales: 2007-2011": 11,
    "England and Wales: 2007 onwards": 12,
    "2020": 13,
    "NO DATA!": -1,
    "INVALID!": -1,
    -1: -1
}

level_dict = {
    "Ground": 0,
    "1st": 1,
    "2nd": 3,
    "3rd": 2,
    "4th": 4,
    "7th": 7,
    "10th": 10,
    "00": 0,
    "01": 1,
    "02": 2,
    "03": 3,
    "0.0": 0,
    "1.0": 1,
    "2.0": 2,
    "4.0": 4,
    "mid floor": 1.5,
    "NODATA!": 12,
    "NO DATA!": -1,
    -1: -1
}

star_dict = {
    "Very Poor": 0,
    "Poor": 1,
    "Average": 2,
    "Good": 3,
    "Very Good": 4,
    "": 6,
    -1: 6
}

epc_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "": -1
}


# ============================================================
# 3. 特征列
# ============================================================

FEATURE_COLS = [
    "bedroom",
    "bathroom",
    "kitchen",
    "lounge",
    "transformation",
    "other",
    "garage",
    "balcony",

    "WALLS_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
    "FLOOR_HEIGHT",
    "NUMBER_HEATED_ROOMS",
    "MAINHEAT_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "ROOF_ENERGY_EFF",

    "shape_factor",
    "constant"
]

TARGET_COL = "EUI"


# ============================================================
# 4. 工具函数
# ============================================================
def save_model_and_get_size(model, model_name, out_dir):
    """
    保存模型，并返回模型文件大小。

    返回：
    model_path: 模型保存路径
    size_bytes: 文件大小，单位 Byte
    size_kb: 文件大小，单位 KB
    size_mb: 文件大小，单位 MB
    """
    model_dir = os.path.join(out_dir, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.joblib")

    # 保存模型
    joblib.dump(model, model_path)

    # 获取模型文件大小
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024

    return model_path, size_bytes, size_kb, size_mb

def safe_map(series, mapper, default=-1):
    """
    字典映射。
    未识别值统一填为 default。
    """
    return series.map(lambda x: mapper.get(x, default))


def check_required_columns(df, cols, df_name):
    """
    检查必要字段是否存在。
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少字段: {missing}")


def expand_image_rows(df):
    """
    将每一条记录扩展为 a-h 八条。
    注意：这里 image_path 不参与模型训练，只用于保持你原脚本的数据结构。
    """
    suffixes = ["a", "b", "c", "d", "e", "f", "g", "h"]
    expanded_rows = []

    for _, row in df.iterrows():
        base_name = os.path.splitext(os.path.basename(str(row["image_path"])))[0]

        for s in suffixes:
            new_row = row.copy()
            new_row["image_path"] = f"{base_name}_{s}.png"
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


def clean_numeric_features(df, feature_cols):
    """
    将特征列转为数值型。
    非法值、无穷值和空值统一处理为 -1。
    """
    df = df.copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(-1)

    return df


def evaluate_predictions(y_true, y_pred):
    """
    返回常用回归指标。
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mean_y = np.mean(y_true)
    cvrmse = rmse / mean_y * 100 if mean_y != 0 else np.nan

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "CVRMSE(%)": cvrmse,
        "R2": r2
    }


def print_metrics(model_name, data_type, metrics):
    """
    打印模型评估结果。
    """
    print("=" * 70)
    print(f"{model_name} | {data_type}评估结果")
    print("=" * 70)
    print(f"{'MAE':<20}: {metrics['MAE']:.4f}")
    print(f"{'MSE':<20}: {metrics['MSE']:.4f}")
    print(f"{'RMSE':<20}: {metrics['RMSE']:.4f}")

    if np.isnan(metrics["CVRMSE(%)"]):
        print(f"{'CVRMSE(%)':<20}: 无法计算")
    else:
        print(f"{'CVRMSE(%)':<20}: {metrics['CVRMSE(%)']:.4f}%")

    print(f"{'R²':<20}: {metrics['R2']:.4f}")
    print("=" * 70)


def measure_predict_time(model, X_test, repeat=5):
    """
    计算测试集调用时间。

    repeat=5 表示重复预测 5 次后取平均。
    这样可以减少单次预测过快导致的计时波动。
    """
    if len(X_test) == 0:
        raise ValueError("X_test 为空，无法计算调用时间。")

    # 预热一次，不计入时间
    _ = model.predict(X_test.iloc[:min(10, len(X_test))])

    times = []
    last_pred = None

    for _ in range(repeat):
        start = time.perf_counter()
        last_pred = model.predict(X_test)
        end = time.perf_counter()
        times.append(end - start)

    avg_total_time = float(np.mean(times))
    avg_single_time = avg_total_time / len(X_test)

    return last_pred, avg_total_time, avg_single_time


# ============================================================
# 5. 数据读取与预处理
# ============================================================

def load_and_prepare_data():
    print("=" * 70)
    print("Step 1. 读取原始数据")
    print("=" * 70)

    data = pd.read_csv(STAT_CSV)
    data_train = pd.read_csv(TRAIN_CSV)
    data_test = pd.read_csv(TEST_CSV)

    check_required_columns(data, ["image_path", "TOTAL_FLOOR_AREA", "FLOOR_HEIGHT", "perimeter"], "统计平面20.csv")
    check_required_columns(data_train, ["image_path", TARGET_COL], "train1.csv")
    check_required_columns(data_test, ["image_path", TARGET_COL], "test1.csv")

    print(f"统计平面数据: {data.shape}")
    print(f"训练集原始数据: {data_train.shape}")
    print(f"测试集原始数据: {data_test.shape}")

    print("\n" + "=" * 70)
    print("Step 2. 处理平面与能耗特征")
    print("=" * 70)

    # 填充基础缺失
    floor_height_mean = pd.to_numeric(data["FLOOR_HEIGHT"], errors="coerce").mean()
    data = data.fillna(-1)

    # 字典映射
    if "CONSTRUCTION_AGE_BAND" in data.columns:
        data["CONSTRUCTION_AGE_BAND"] = safe_map(data["CONSTRUCTION_AGE_BAND"], age_dict)

    energy_eff_cols = [
        "WALLS_ENERGY_EFF",
        "WINDOWS_ENERGY_EFF",
        "HOT_WATER_ENERGY_EFF",
        "LIGHTING_ENERGY_EFF",
        "MAINHEAT_ENERGY_EFF",
        "ROOF_ENERGY_EFF"
    ]

    for col in energy_eff_cols:
        if col in data.columns:
            data[col] = safe_map(data[col], star_dict)

    # FLOOR_HEIGHT 处理
    data["FLOOR_HEIGHT"] = pd.to_numeric(data["FLOOR_HEIGHT"], errors="coerce")
    data.loc[data["FLOOR_HEIGHT"] == -1, "FLOOR_HEIGHT"] = floor_height_mean
    data["FLOOR_HEIGHT"] = data["FLOOR_HEIGHT"].fillna(floor_height_mean)

    # 面积字段避免除以 0
    data["TOTAL_FLOOR_AREA"] = pd.to_numeric(data["TOTAL_FLOOR_AREA"], errors="coerce")
    data.loc[data["TOTAL_FLOOR_AREA"] <= 0, "TOTAL_FLOOR_AREA"] = np.nan

    outline_cols = [
        "bathroom_outline",
        "kitchen_outline",
        "transformation_outline",
        "other_outline",
        "balcony_outline",
        "garage_outline"
    ]

    for col in outline_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col] / data["TOTAL_FLOOR_AREA"]

    # base_image_path 去重
    data["image_path"] = data["image_path"].astype(str)
    data["base_image_path"] = data["image_path"].apply(
        lambda x: os.path.splitext(x)[0][:-2]
    )

    data = data.drop_duplicates(subset=["base_image_path"])

    # shape factor
    data["perimeter"] = pd.to_numeric(data["perimeter"], errors="coerce")
    data["shape_factor"] = data["perimeter"] / data["TOTAL_FLOOR_AREA"]

    # 辅助形态指标
    for col in outline_cols:
        if col not in data.columns:
            data[col] = 0

    data["aux_shape_factor"] = (
        data["bathroom_outline"].fillna(0)
        + data["kitchen_outline"].fillna(0)
        + data["balcony_outline"].fillna(0)
        + data["garage_outline"].fillna(0)
    )

    data["constant"] = 1
    data["image_path"] = data["image_path"].apply(lambda x: os.path.basename(str(x)))

    # 这里保留原脚本需要合并的平面特征
    keep_cols = [
        "image_path",
        "bedroom",
        "bathroom",
        "kitchen",
        "lounge",
        "transformation",
        "other",
        "garage",
        "balcony",
        "shape_factor",
        "aux_shape_factor",
        "constant"
    ]

    for col in keep_cols:
        if col not in data.columns:
            data[col] = -1

    data = data[keep_cols]

    # 统一 image_path 格式
    data_train["image_path"] = data_train["image_path"].apply(lambda x: os.path.basename(str(x)))
    data_test["image_path"] = data_test["image_path"].apply(lambda x: os.path.basename(str(x)))

    # 合并
    train_data = pd.merge(data_train, data, on="image_path", how="inner")
    test_data = pd.merge(data_test, data, on="image_path", how="inner")

    print(f"合并后训练集: {train_data.shape}")
    print(f"合并后测试集: {test_data.shape}")

    if train_data.empty:
        raise ValueError("训练集合并后为空。请检查 train1.csv 与 统计平面20.csv 中 image_path 是否一致。")

    if test_data.empty:
        raise ValueError("测试集合并后为空。请检查 test1.csv 与 统计平面20.csv 中 image_path 是否一致。")

    # 扩展 a-h
    train_data = expand_image_rows(train_data)
    test_data = expand_image_rows(test_data)

    print(f"扩展后训练集: {train_data.shape}")
    print(f"扩展后测试集: {test_data.shape}")

    # 检查模型字段
    check_required_columns(train_data, FEATURE_COLS + [TARGET_COL], "训练集")
    check_required_columns(test_data, FEATURE_COLS + [TARGET_COL], "测试集")

    # 数值清洗
    train_data = clean_numeric_features(train_data, FEATURE_COLS)
    test_data = clean_numeric_features(test_data, FEATURE_COLS)

    train_data[TARGET_COL] = pd.to_numeric(train_data[TARGET_COL], errors="coerce")
    test_data[TARGET_COL] = pd.to_numeric(test_data[TARGET_COL], errors="coerce")

    train_data = train_data.dropna(subset=[TARGET_COL])
    test_data = test_data.dropna(subset=[TARGET_COL])

    print(f"去除目标变量空值后训练集: {train_data.shape}")
    print(f"去除目标变量空值后测试集: {test_data.shape}")

    return train_data, test_data


# ============================================================
# 6. 构建模型
# ============================================================

def build_models():
    models = {}

    # 1. 线性回归
    models["LinearRegression"] = LinearRegression()

    # 2. Random Forest
    models["RandomForest"] = RandomForestRegressor()

    # 3. XGBoost
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor()

    except ImportError:
        print("警告：当前环境未安装 xgboost，已跳过 XGBoost。")
        print("如需使用，请先运行：pip install xgboost")

    return models


# ============================================================
# 7. 训练、计时与评估
# ============================================================

def train_and_evaluate_all_models(train_data, test_data):
    X_train = train_data[FEATURE_COLS]
    y_train = train_data[TARGET_COL].values

    X_test = test_data[FEATURE_COLS]
    y_test = test_data[TARGET_COL].values

    models = build_models()

    summary_rows = []

    for model_name, model in models.items():
        print("\n\n" + "#" * 80)
        print(f"开始训练模型: {model_name}")
        print("#" * 80)

        # -------------------------
        # 训练时间
        # -------------------------
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_end = time.perf_counter()

        train_time = train_end - train_start

        print(f"\n{model_name} 训练完成")
        print(f"训练时间: {train_time:.6f} 秒")

        # -------------------------
        # 保存模型并统计模型大小
        # -------------------------
        model_path, model_size_bytes, model_size_kb, model_size_mb = save_model_and_get_size(
            model=model,
            model_name=model_name,
            out_dir=OUT_DIR
        )

        print(f"模型已保存: {model_path}")
        print(f"模型文件大小: {model_size_bytes} Byte")
        print(f"模型文件大小: {model_size_kb:.4f} KB")
        print(f"模型文件大小: {model_size_mb:.6f} MB")

        # -------------------------
        # 训练集评估
        # -------------------------
        train_pred = model.predict(X_train)
        train_metrics = evaluate_predictions(y_train, train_pred)
        print_metrics(model_name, "训练集", train_metrics)

        # -------------------------
        # 测试集调用时间
        # -------------------------
        test_pred, test_total_call_time, test_single_call_time = measure_predict_time(
            model=model,
            X_test=X_test,
            repeat=5
        )

        print(f"{model_name} 测试集调用时间统计")
        print("-" * 70)
        print(f"测试集样本数: {len(X_test)}")
        print(f"测试集平均总调用时间: {test_total_call_time:.8f} 秒")
        print(f"测试集单样本平均调用时间: {test_single_call_time:.10f} 秒/样本")
        print("-" * 70)

        # -------------------------
        # 测试集评估
        # -------------------------
        test_metrics = evaluate_predictions(y_test, test_pred)
        print_metrics(model_name, "测试集", test_metrics)

        # -------------------------
        # 保存预测结果
        # -------------------------
        pred_df = test_data[["image_path"]].copy()
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = test_pred
        pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
        pred_df["abs_error"] = pred_df["error"].abs()

        pred_path = os.path.join(OUT_DIR, f"{model_name}_test_predictions.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        # -------------------------
        # 汇总结果
        # -------------------------
        summary_rows.append({
            "Model": model_name,

            "Train_Time_s": train_time,
            "Test_Total_Call_Time_s": test_total_call_time,
            "Test_Single_Call_Time_s": test_single_call_time,

            "Model_Size_Byte": model_size_bytes,
            "Model_Size_KB": model_size_kb,
            "Model_Size_MB": model_size_mb,
            "Model_File": model_path,

            "Train_MAE": train_metrics["MAE"],
            "Train_MSE": train_metrics["MSE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_CVRMSE_percent": train_metrics["CVRMSE(%)"],
            "Train_R2": train_metrics["R2"],

            "Test_MAE": test_metrics["MAE"],
            "Test_MSE": test_metrics["MSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_CVRMSE_percent": test_metrics["CVRMSE(%)"],
            "Test_R2": test_metrics["R2"],

            "Prediction_File": pred_path
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(OUT_DIR, "model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n\n" + "=" * 80)
    print("所有模型对比结果")
    print("=" * 80)
    print(summary_df)

    print("\n结果已保存:")
    print(summary_path)

    return summary_df


# ============================================================
# 8. 主程序
# ============================================================

def main():
    train_data, test_data = load_and_prepare_data()
    summary_df = train_and_evaluate_all_models(train_data, test_data)


if __name__ == "__main__":
    main()