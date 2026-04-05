"""
Training utilities for all audio classification models.

Supports FHE-aware noise injection for improved encrypted inference accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader


def train_model(X, y, model, epochs=200, lr=1e-3, val_split=0.2,
                batch_size=32, seed=42, noise_std=0.0):
    """
    Train a classification model.

    Args:
        X: np.ndarray of shape (n, input_dim) — feature vectors.
        y: np.ndarray of shape (n,) — labels.
        model: orion.nn.Module instance.
        epochs: Number of training epochs.
        lr: Learning rate.
        val_split: Fraction of data for validation.
        batch_size: Batch size.
        seed: Random seed.
        noise_std: Gaussian noise std added to features during training.
            Simulates FHE precision loss for wider decision margins.

    Returns:
        model: Trained model (best validation accuracy state).
        scaler: Fitted StandardScaler.
        metrics: dict with train_acc, val_acc, best_epoch.
        X_val: Validation features (scaled).
        y_val: Validation labels.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Train/val split
    n_val = int(len(X_scaled) * val_split)
    indices = np.random.permutation(len(X_scaled))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_val, y_val = X_scaled[val_idx], y[val_idx]

    train_X = torch.tensor(X_train)
    train_y = torch.tensor(y_train)
    val_X = torch.tensor(X_val)
    val_y = torch.tensor(y_val)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=30, factor=0.5, verbose=False)

    best_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 80

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # FHE-aware noise injection during training
            if noise_std > 0:
                noise = torch.randn_like(batch_X) * noise_std
                batch_X = batch_X + noise

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_out = model(val_X)
            val_pred = val_out.argmax(dim=1).numpy()
            val_acc = accuracy_score(val_y.numpy(), val_pred)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {early_stop_patience} epochs)")
            break

    model.load_state_dict(best_state)

    # Final train accuracy
    model.eval()
    with torch.no_grad():
        train_out = model(train_X)
        train_pred = train_out.argmax(dim=1).numpy()
        train_acc = accuracy_score(y_train, train_pred)

    metrics = {
        "train_acc": train_acc,
        "val_acc": best_acc,
        "best_epoch": best_epoch,
    }

    return model, scaler, metrics, X_val, y_val
