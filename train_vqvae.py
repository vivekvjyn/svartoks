import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from libs.utils import *
from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset(data, max_length, split_ratio):
    """
    Prepare the dataset for training.

    Args:
        data (list): The data to prepare.
        max_length (int): The maximum length of the sequences.
        split_ratio (float): The ratio to split the data.

    Returns:
        tuple: The left and right sequences.
    """
    seqs = [replace_nans(x) for sample in data for x in (sample[2], sample[3], sample[4])]

    split_index = int(len(seqs) * split_ratio)
    left_seqs = np.array([pad(seq, max_length) for seq in seqs[split_index:]])
    right_seqs = np.array([pad(seq, max_length) for seq in seqs[:split_index]])

    left_seqs = torch.tensor(left_seqs, dtype=torch.float).to(device)
    right_seqs = torch.tensor(right_seqs, dtype=torch.float).to(device)

    return left_seqs, right_seqs


def propagate(model, optimizer, data_loader, mode):
    """
    Propagate the model through the data loader.

    Args:
        model (torch.nn.Module): The model to propagate.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        data_loader (torch.utils.data.DataLoader): The data loader to use.
        mode (str): The mode to use.

    Returns:
        float: The total loss.
    """
    model.train() if mode == 'train' else model.eval()
    total_loss = 0

    for (batch,) in tqdm(data_loader):
        batch = batch.to(dtype=torch.float, device=device)
        loss, *_ = model.shared_eval(batch, optimizer, mode)
        total_loss += loss.item()

    return total_loss


def fit_model(train_loader, val_loader, test_loader, epochs, run_id):
    """
    Train the VQ-VAE tokenizer.

    Args:
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        epochs (int): Number of training epochs.
        run_id (str): Unique identifier for the training run.
    """

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    history = pd.DataFrame(columns=["train loss", "val loss"])
    min_val_loss = np.inf

    print("Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        train_loss = propagate(model, optimizer, train_loader, 'train')
        val_loss = propagate(model, optimizer, val_loader, 'test')

        log_metrics(train_loss, mode="Train")
        log_metrics(val_loss, mode="Validation")
        history.loc[len(history)] = [train_loss, val_loss]

        if val_loss < min_val_loss:
            save_model(model, run_id)
            min_val_loss = val_loss

    model = torch.load(f'saved_models/{run_id}/tokenizer.pth', map_location=device, weights_only=False)
    print("Evaluating")
    test_loss = propagate(model, optimizer, test_loader, 'test')
    print(f"Test Loss: {test_loss:.8f}")

    save_logs(history, test_loss, run_id)


def get_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train VQ-VAE tokenizer")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size")
    parser.add_argument('--run_id', type=str, default='newrun1', help="Experiment identifier")

    return parser.parse_args()


def main():
    args = get_args()

    train_data, test_data = load_datasets()

    train, val = prepare_dataset(train_data, args.max_length, split_ratio=0.25)
    test, _ = prepare_dataset(test_data, args.max_length, split_ratio=0)

    train_loader = DataLoader(TensorDataset(train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test), batch_size=args.batch_size, shuffle=True)

    fit_model(train_loader, val_loader, test_loader, args.epochs, args.run_id)


if __name__ == '__main__':
    main()
