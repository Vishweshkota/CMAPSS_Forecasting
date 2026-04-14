from pathlib import Path
import joblib
import torch
import torch.nn as nn


# LSTM model architecture used during training.
# We recreate this class so saved PyTorch weights can be loaded for inference.
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Use the final hidden state to predict one RUL value.
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        output = self.fc(last_hidden)
        return output


# Project root and saved model locations.
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "saved_models"

# Use GPU if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the saved linear regression model.
def load_linear_regression_model():
    model_path = MODEL_DIR / "final_linear_regression.joblib"
    return joblib.load(model_path)


# Load the saved random forest model.
def load_random_forest_model():
    model_path = MODEL_DIR / "final_random_forest.joblib"
    return joblib.load(model_path)


# Rebuild the LSTM architecture and load saved weights + metadata.
def load_lstm_model():
    metadata_path = MODEL_DIR / "final_lstm_optimized_metadata.joblib"
    model_path = MODEL_DIR / "final_lstm_optimized.pth"

    metadata = joblib.load(metadata_path)

    model = LSTMRegressor(
        input_size=metadata["input_size"],
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
        dropout=metadata["dropout"]
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata


# Load preprocessing objects needed to transform raw sensor rows.
def load_preprocessing_artifacts():
    artifact_path = MODEL_DIR / "preprocessing_artifacts.joblib"
    return joblib.load(artifact_path)


# Convenience loader to initialize everything at app startup.
def load_all_assets():
    lr_model = load_linear_regression_model()
    rf_model = load_random_forest_model()
    lstm_model, lstm_metadata = load_lstm_model()
    preprocessing_artifacts = load_preprocessing_artifacts()

    return {
        "device": DEVICE,
        "lr_model": lr_model,
        "rf_model": rf_model,
        "lstm_model": lstm_model,
        "lstm_metadata": lstm_metadata,
        "preprocessing_artifacts": preprocessing_artifacts
    }
