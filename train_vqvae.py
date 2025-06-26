import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from libs.utils import *
from models.vqvae import VQVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_tensor(ts, max_length):
    padded = np.array([pad(x, max_length) for x in ts])

    padded_tensor = torch.tensor(padded, dtype=torch.float).to(DEVICE)

    return padded_tensor


def process_data(data, max_length, validation=False):
    ts = list()

    for svara, form, prec, curr, succ in data:
        ts.append(replace_nans(prec))
        ts.append(replace_nans(curr))
        ts.append(replace_nans(prec))

    if validation:
        ts_train = create_tensor(ts[len(ts) // 4:], max_length)
        ts_val = create_tensor(ts[:len(ts) // 4], max_length)
        return ts_train, ts_val
    else:
        ts = create_tensor(ts, max_length)
        return ts


def create_data_loader(ts, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(ts)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(tokenizer, optimizer, loader, mode):
    epoch_loss = 0

    for (ts,) in tqdm(loader):
        ts = ts.to(dtype=torch.float, device=DEVICE)
        loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings = \
            tokenizer.shared_eval(ts, optimizer, 'train')

        epoch_loss += loss.item()

    return epoch_loss


def write_results(history, test_loss, run_id):
    os.makedirs(f'results/{run_id}', exist_ok=True)

    df = pd.DataFrame(history)
    df.to_csv(f'results/{run_id}/training_log.csv', index=False)

    df = pd.DataFrame([{'test_loss': test_loss}])
    df.to_csv(f'results/{run_id}/test_log.csv', index=False)


def train(train_loader, val_loader, test_loader, epochs, run_id):
    history = list()
    tokenizer = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=0.01)

    print("Training...")
    for epoch in range(epochs):
        tokenizer.train()
        train_loss = run_epoch(tokenizer, optimizer, train_loader, 'train') / len(train_loader)

        tokenizer.eval()
        val_loss = run_epoch(tokenizer, optimizer, val_loader, 'test') / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.8f} Val Loss = {val_loss:.8f}")

        history.append({"epoch": epoch + 1, "train loss": train_loss, "val loss": val_loss})

    print('Testing...')
    tokenizer.eval()
    test_loss = run_epoch(tokenizer, optimizer, test_loader, 'test') / len(test_loader)

    print(f"Test Loss: {test_loss:.8f}")

    write_results(history, test_loss, run_id)

    os.makedirs(f'saved_models/{run_id}', exist_ok=True)
    torch.save(tokenizer, os.path.join(f'saved_models/{run_id}/tokenizer.pth'))
    print(f"Model saved to saved_models/{run_id}/tokenizer.pth")


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer with dataset")
    parser.add_argument('--max_length', type=int, default=1024, help="Maximum sequence length")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--run_id', type=int, default='0001', help="Batch size for training")
    args = parser.parse_args()

    print('Loading data..')

    with open('dataset/TRAIN.pkl', 'rb') as file:
        train_data = pickle.load(file)

    with open('dataset/PRETRAIN_TRAIN.pkl', 'rb') as file:
        train_data += pickle.load(file)

    with open('dataset/TEST.pkl', 'rb') as file:
        test_data = pickle.load(file)

    print('Loading finished')

    print('Processing data..')

    ts_train, ts_val = process_data(train_data, args.max_length, validation=True)
    ts_test = process_data(test_data, args.max_length)

    train_loader = create_data_loader(ts_train, args.batch_size)
    val_loader = create_data_loader(ts_val, args.batch_size)
    test_loader = create_data_loader(ts_test, args.batch_size)

    print('Processing finished')

    train(train_loader, val_loader, test_loader, args.epochs, args.run_id)


if __name__ == '__main__':
    main()
