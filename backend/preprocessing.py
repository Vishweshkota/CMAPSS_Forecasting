import pandas as pd


# Validate that one raw incoming row contains all required fields.
def validate_raw_row(raw_row, artifacts):
    required_cols = ["tag", "engine_id", "time_cycles"] + artifacts["setting_cols"] + artifacts["sensor_cols"]

    missing_cols = [col for col in required_cols if col not in raw_row]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    tag = raw_row["tag"]
    if tag not in artifacts["datasets"]:
        raise ValueError(f"Unsupported tag: {tag}")

    return True


# Assign operating condition using either fixed value or saved KMeans model.
def assign_op_condition(row_df, dataset_cfg, setting_cols):
    if dataset_cfg["op_condition_mode"] == "single":
        row_df["op_condition"] = dataset_cfg["op_condition_value"]
    elif dataset_cfg["op_condition_mode"] == "kmeans":
        kmeans_model = dataset_cfg["kmeans_model"]
        row_df["op_condition"] = kmeans_model.predict(row_df[setting_cols])[0]
    else:
        raise ValueError(f"Unsupported op_condition_mode: {dataset_cfg['op_condition_mode']}")

    return row_df


# Normalize all sensor columns using condition-specific mean and std.
def normalize_sensors(row_df, dataset_cfg, sensor_cols):
    op_condition = row_df["op_condition"].iloc[0]
    condition_means = dataset_cfg["condition_means"]
    condition_stds = dataset_cfg["condition_stds"]

    for sensor in sensor_cols:
        mean_val = condition_means.loc[op_condition, sensor]
        std_val = condition_stds.loc[op_condition, sensor]
        row_df[sensor] = (row_df[sensor] - mean_val) / std_val

    return row_df


# Full preprocessing pipeline for one raw sensor row.
# Returns both a debug-friendly row and the final model-ready feature row.
def preprocess_single_row(raw_row, artifacts):
    validate_raw_row(raw_row, artifacts)

    tag = raw_row["tag"]
    dataset_cfg = artifacts["datasets"][tag]

    setting_cols = artifacts["setting_cols"]
    sensor_cols = artifacts["sensor_cols"]
    feature_cols = artifacts["feature_cols"]

    row_df = pd.DataFrame([raw_row])

    row_df = assign_op_condition(row_df, dataset_cfg, setting_cols)
    row_df = normalize_sensors(row_df, dataset_cfg, sensor_cols)

    processed_row = row_df[feature_cols].copy()

    return {
        "raw_with_op_condition": row_df,
        "processed_row": processed_row
    }
