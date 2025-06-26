import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from libs.utils import *
from models.vqvae import VQVAE
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 1024
EPOCHS = 2000
BATCH_SIZE = 64

def process_data(data, max_length, validation=False):

    def create_tensor(ts):
        padded = torch.tensor(np.array([pad(x, max_length) for x in ts]), dtype=torch.float).to(DEVICE)
        return padded

    ts = list()
    for svara, form, prec, curr, succ in data:
        ts.append(replace_nans(prec))
        ts.append(replace_nans(curr))
        ts.append(replace_nans(prec))

    if validation:
        ts_train = create_tensor(ts[len(ts) // 4:])
        ts_val = create_tensor(ts[:len(ts) // 4])

        return ts_train, ts_val
    else:
        return create_tensor(ts)

def create_data_loader(ts, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(ts)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_tokenizer(train_loader, val_loader, test_loader, epochs):
    tokenizer = VQVAE().to(DEVICE)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=0.01)
    for epoch in range(epochs):

        tokenizer.train()
        train_loss = 0
        train_total = 0
        for (ts,) in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            ts = ts.to(dtype=torch.float, device=DEVICE)
            loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings = \
                tokenizer.shared_eval(ts, optimizer, 'train')

            train_loss += recon_error.item()
            train_total += 1

        tokenizer.eval()
        val_loss = 0
        val_total = 0
        for (ts,) in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
            ts = ts.to(dtype=torch.float, device=DEVICE)
            loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings = \
                tokenizer.shared_eval(ts, optimizer, 'test')

            val_loss += recon_error.item()
            val_total += 1

        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.8f} Val Loss = {avg_val_loss:.8f}")

    print('Testing')
    tokenizer.eval()
    test_loss = 0
    test_total = 0

    for (ts,) in tqdm(test_loader):
        ts = ts.to(dtype=torch.float, device=DEVICE)
        loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings = \
            tokenizer.shared_eval(ts, optimizer, 'test')

        test_loss += recon_error.item()
        test_total += 1

    avg_test_loss = test_loss / test_total

    print(f"Test Loss: {avg_test_loss:.8f}")

    torch.save(tokenizer, os.path.join(f'saved_models/tokenizer.pth'))


def main():
    print('Loading data..')

    with open('dataset/TRAIN.pkl', 'rb') as file:
        train_data = pickle.load(file)

    with open('dataset/PRETRAIN_TRAIN.pkl', 'rb') as file:
        train_data += pickle.load(file)

    with open('dataset/TEST.pkl', 'rb') as file:
        test_data = pickle.load(file)

    print('Loading finished..')

    ts_train, ts_val = process_data(train_data, MAX_LENGTH, validation=True)
    ts_test = process_data(test_data, MAX_LENGTH)

    train_loader = create_data_loader(ts_train, BATCH_SIZE)
    val_loader = create_data_loader(ts_val, BATCH_SIZE)
    test_loader = create_data_loader(ts_test, BATCH_SIZE)

    print("Training tokenizer")
    train_tokenizer(train_loader, val_loader, test_loader, EPOCHS)

if __name__ == '__main__':
    main()
