import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from libs.utils import pad, replace_nans
from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def padded_tensor(sequences, max_length):
    padded = np.array([pad(seq, max_length) for seq in sequences])
    return torch.tensor(padded, dtype=torch.float).to(device)


def process_dataset(data, max_length, validation=False):
    sequences = [replace_nans(x) for sample in data for x in (sample[2], sample[3], sample[4])]

    if validation:
        split_index = len(sequences) // 4
        val_sequences = padded_tensor(sequences[:split_index], max_length)
        train_sequences = padded_tensor(sequences[split_index:], max_length)
        return train_sequences, val_sequences
    else:
        return padded_tensor(sequences, max_length)


def build_data_loader(tensor_data, batch_size, shuffle=True):
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, optimizer, data_loader, mode):
    model.train() if mode == 'train' else model.eval()
    total_loss = 0

    for (batch,) in data_loader:
        batch = batch.to(dtype=torch.float, device=device)
        loss, *_ = model.shared_eval(batch, optimizer, mode)
        total_loss += loss.item()

    return total_loss / len(data_loader)


def save_logs(history, test_loss, run_id):
    result_dir = f'results/{run_id}'
    os.makedirs(result_dir, exist_ok=True)

    pd.DataFrame(history).to_csv(f'{result_dir}/training_log.csv', index=False)
    pd.DataFrame([{'test_loss': test_loss}]).to_csv(f'{result_dir}/test_log.csv', index=False)


def save_model(model, run_id):
    model_dir = f'saved_models/{run_id}'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, os.path.join(model_dir, 'tokenizer.pth'))
    print(f"Model saved to {model_dir}/tokenizer.pth")


def train_model(train_loader, val_loader, test_loader, epochs, run_id):
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = []
    min_val_loss = np.inf

    print("Training...")
    for epoch in range(epochs):
        train_loss = run_epoch(model, optimizer, train_loader, 'train')
        val_loss = run_epoch(model, optimizer, val_loader, 'test')

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.8f}, Val Loss = {val_loss:.8f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < min_val_loss:
            save_model(model, run_id)
            min_val_loss = val_loss

    print("Evaluating")
    test_loss = run_epoch(model, optimizer, test_loader, 'test')
    print(f"Test Loss: {test_loss:.8f}")

    save_logs(history, test_loss, run_id)



def load_datasets():
    print("Loading datasets...")
    with open('dataset/TRAIN.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('dataset/PRETRAIN_TRAIN.pkl', 'rb') as f:
        train_data += pickle.load(f)
    with open('dataset/TEST.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print("Datasets loaded.")
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE tokenizer")
    parser.add_argument('--max_length', type=int, default=1024, help="Maximum sequence length")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--run_id', type=str, default='0001', help="Experiment identifier")

    args = parser.parse_args()

    train_data, test_data = load_datasets()

    train_tensor, val_tensor = process_dataset(train_data, args.max_length, validation=True)
    test_tensor = process_dataset(test_data, args.max_length)

    train_loader = build_data_loader(train_tensor, args.batch_size)
    val_loader = build_data_loader(val_tensor, args.batch_size)
    test_loader = build_data_loader(test_tensor, args.batch_size)

    train_model(train_loader, val_loader, test_loader, args.epochs, args.run_id)


if __name__ == '__main__':
    main()
