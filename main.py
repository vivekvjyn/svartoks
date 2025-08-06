import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import sklearn.model_selection
from models.vqvae import VQVAE
from libs.data import *
from libs.io import *
from libs.logger import Logger

parser = argparse.ArgumentParser(description="Tokenized Time Series Embeddings")
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--num_embeddings', type=int, default=512)
parser.add_argument('--run_id', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = Logger(args.run_id)

def totem_dataset(data):
    """
    Prepare the dataset for training.

    Args:
        data (list): The data to prepare.
        max_len (int): The maximum length of the sequences.

    Returns:
        seqs: Sequences.
    """
    seqs = [np.nan_to_num(np.array(ts) / 2400, nan=-1).tolist() for row in data for ts in (row[2], row[3], row[4])]
    seqs = np.array([np.pad(seq, (0, max(args.max_len - len(seq), 0)), constant_values=0)[:args.max_len] for seq in seqs], dtype=np.float32)
    seqs = torch.from_numpy(seqs).unsqueeze(1).to(device)
    return seqs

def propagate(tokenizer, optimizer, criterion, data_loader, back_prop=False):
    """Propagate the tokenizer through the data loader.

    Args:
        tokenizer (nn.Module): The tokenizer to propagate.
        optimizer (Optimizer): The optimizer to use for parameter updates.
        criterion (nn.Module): The loss function to use.
        data_loader (DataLoader): The data loader to use for loading the data.
        back_prop (bool): Whether to perform backpropagation.

    Returns:
        float: The total loss.
    """
    tokenizer.train() if back_prop else tokenizer.eval()
    total_loss = 0
    for i, (seq,) in enumerate(data_loader):
        logger.progress_bar(i + 1, len(data_loader))
        recon = tokenizer(seq)
        loss = criterion(recon, seq.squeeze(1))
        if back_prop:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    total_loss = total_loss / len(data_loader)
    return total_loss

def train(tokenizer, optimizer, criterion, scheduler, train_loader, val_loader, history):
    """Train the tokenizer using the given data loaders and hyperparameters.

    Args:
        tokenizer (nn.Module): The tokenizer to train.
        optimizer (Optimizer): The optimizer to use for parameter updates.
        criterion (Criterion): The loss criterion to use.
        scheduler (Scheduler): The learning rate scheduler to use.
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        history (DataFrame): The history of training and validation metrics.
    """
    min_val_loss = np.inf
    for epoch in range(args.epochs):
        logger.message(f"Epoch {epoch+1}/{args.epochs}:")
        train_loss = propagate(tokenizer, optimizer, criterion, train_loader, back_prop=True)
        log_metrics(train_loss, mode="Train", logger=logger)
        val_loss = propagate(tokenizer, optimizer, criterion, val_loader, back_prop=False)
        scheduler.step(val_loss)
        log_metrics(val_loss, mode="Validation", logger=logger)
        if val_loss < min_val_loss:
            save_tokenizer(tokenizer, args.run_id, logger=logger)
            min_val_loss = val_loss
        history.loc[len(history)] = [train_loss, val_loss]

def fit_tokenizer(train_loader, val_loader):
    """Train a model on a given dataset

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_classes (int): Number of classes in the dataset.
        fold (int): Fold number for cross-validation.
        logger (Logger): Logger object for logging.

    Returns:
        tuple: Maximum training F1 score and maximum validation F1 score.
    """
    tokenizer = VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=4)
    history = pd.DataFrame(columns=["train loss", "val loss"])
    train(tokenizer, optimizer, criterion, scheduler, train_loader, val_loader, history)
    save_history(history, args.run_id)

def test_tokenizer(data_loader, classes, run_id, logger):
    """
    Test a model on a given dataset

    Args:
        data_loader (DataLoader): Data loader for the test dataset.
        classes (list): Class labels.
        run_id (str): ID of the run.
        logger (Logger): Logger object for logging.

    Returns:
        float: Test set F1 score.
    """
    tokenizer = VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim).to(device)
    state_dict = torch.load(f'checkpoints/{run_id}/tokenizer.pth', map_location=device)
    tokenizer.load_state_dict(state_dict)
    criterion = torch.nn.MSELoss()
    loss = propagate(tokenizer, None, criterion, data_loader, back_prop=False)
    log_metrics(loss, mode='Test', logger=logger)

def main():
    train_data, test_data, classes = load_datasets()

    train_dataset = torch.utils.data.TensorDataset(totem_dataset(train_data))
    test_dataset = torch.utils.data.TensorDataset(totem_dataset(test_data))

    logger.heading('TRAINING MODEL')
    train_dataset, val_dataset = sklearn.model_selection.train_test_split(train_dataset, test_size=0.2, random_state=42)
    fit_tokenizer(torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size), torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size))
    train_curve(args.run_id, logger=logger)

    logger.heading('TEST SET EVALUATION')
    test_tokenizer(torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size), classes, args.run_id, logger)


if __name__ == '__main__':
    main()
