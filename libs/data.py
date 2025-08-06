import pickle
from typing_extensions import Sequence
import numpy as np
import torch

def column(dataset, idx):
    """Obtain a column from a list of lists.

    Args:
        dataset (list): The list of lists.
        idx (int): The index of the column to obtain.

    Returns:
        list: The column.
    """
    assert idx in range(2, 5), f"column {idx} is not a sequence"
    return [np.nan_to_num(sample[idx], nan=-4800).tolist() for sample in dataset]

def features(seqs, max_len, device):
    """Obtain padded sequences and masks.

    Args:
        seqs (list): Sequences.
        max_len (int): Length of the sequences.
        device (torch.device): The device to use.

    Returns:
        tuple: The padded sequences and masks.
    """
    seqs = torch.tensor(np.array([np.pad(seq, (0, max(max_len - len(seq), 0)), constant_values=0)[:max_len] for seq in seqs]), dtype=torch.float, device=device)
    masks = torch.tensor([[False] * len(seq) + [True] * (max_len - len(seq)) for seq in seqs], dtype=torch.float, device=device)
    return seqs, masks

def load_datasets(use_forms=False, pretrain=False):
    """Load datasets.

    Args:
        svara_forms (bool): Whether to load svara form labels.
        pretrain (bool): Whether to load pretraining datasets.

    Returns:
        tuple: The loaded datasets.
    """
    with open(f"dataset/{'PRETRAIN_TRAIN' if pretrain else 'TRAIN'}.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(f"dataset/{'PRETRAIN_TEST' if pretrain else 'TEST'}.pkl" , 'rb') as f:
        test_data = pickle.load(f)
    with open(f"dataset/LABELS.pkl" , 'rb') as f:
        labels = pickle.load(f)
        classes = labels[f"{'svara_form' if use_forms else 'svara'}"].values()
    return train_data, test_data, classes
