import os
import pickle
import numpy as np
import pandas as pd
import torch

def load_datasets():
    """
    Load datasets from pickle files.

    Returns:
        tuple: Train and test datasets.
    """
    print("Loading datasets...")
    with open('dataset/PRETRAIN_TRAIN.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('dataset/TEST.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print("Datasets loaded.")

    return train_data, test_data


def pad(ts, max_length):
    """
    Pad time series to a given length.

    Args:
        ts (list): Time series to pad.
        max_length (int): Maximum length of the time series.

    Returns:
        list: Padded time series.
    """
    if len(ts) > max_length:
        return ts[:max_length]
    else:
        return np.pad(ts, (0, max_length - len(ts)), mode='constant', constant_values=-2)


def create_mask(lengths, max_length):
    """
    Create masks for given lengths.

    Args:
        lengths (list): Lengths of the time series.
        max_length (int): Maximum length of the time series.

    Returns:
        np.ndarray: Array of masks.
    """
    masks = [[False] * length + [True] * (max_length - length) for length in lengths]
    return np.array(masks)


def replace_nans(ts):
    """
    Replace NaN values in a time series with a constant value.

    Args:
        ts (list): Time series to process.

    Returns:
        list: Processed time series.
    """
    ts = np.array(ts, dtype=np.float32)
    ts[np.isnan(ts)] = -4800
    ts = ts / 2400.0
    return ts.tolist()


def log_metrics(loss, mode):
    """
    Log evaluation metrics.

    Args:
        loss (float): Loss value.
        mode (str): Mode of operation ('train' or 'val').
    """
    print(f"\t{mode} Loss: {loss:.4f}")


def save_logs(history, test_loss, run_id):
    """
    Save evaluation logs to csv.

    Args:
        history (pd.DataFrame): Training history.
        test_loss (float): Test loss.
        run_id (str): Run identifier.
    """
    result_dir = f'results/{run_id}'
    os.makedirs(result_dir, exist_ok=True)

    history.to_csv(f'{result_dir}/training_log.csv', index=False)


def save_model(model, run_id):
    """
    Save model to disk.

    Args:
        model (torch.nn.Module): Model to save.
        run_id (str): Run identifier.
    """
    model_dir = f'saved_models/{run_id}'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, os.path.join(model_dir, 'tokenizer.pth'))
    print(f"Model saved to {model_dir}/tokenizer.pth")
