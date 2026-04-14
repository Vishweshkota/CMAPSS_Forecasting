import numpy as np
import torch


# Convert a processed rolling window into flat input for LR and RF.
def prepare_flat_window(window):
    return window.reshape(1, -1)


# Convert a processed rolling window into 3D tensor input for LSTM.
def prepare_lstm_window(window, device):
    window_array = np.array(window, dtype=np.float32).reshape(1, window.shape[0], window.shape[1])
    return torch.tensor(window_array, dtype=torch.float32).to(device)


# Predict RUL using the linear regression model.
def predict_with_lr(window, lr_model):
    flat_input = prepare_flat_window(window)
    prediction = lr_model.predict(flat_input)[0]
    return float(prediction)


# Predict RUL using the random forest model.
def predict_with_rf(window, rf_model):
    flat_input = prepare_flat_window(window)
    prediction = rf_model.predict(flat_input)[0]
    return float(prediction)


# Predict RUL using the LSTM model.
def predict_with_lstm(window, lstm_model, device):
    lstm_input = prepare_lstm_window(window, device)

    with torch.no_grad():
        prediction = lstm_model(lstm_input).cpu().numpy().ravel()[0]

    return float(prediction)


# Unified prediction entry point for all supported models.
def predict_rul(model_name, window, assets):
    if model_name == "lr":
        return predict_with_lr(window, assets["lr_model"])

    elif model_name == "rf":
        return predict_with_rf(window, assets["rf_model"])

    elif model_name == "lstm":
        return predict_with_lstm(window, assets["lstm_model"], assets["device"])

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
