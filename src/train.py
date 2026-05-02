import torch
import torch.nn as nn
import torch.optim as optim

from src.config import EPOCHS, PATIENCE


def train_model(model, train_loader, val_loader,
                save_path: str, device: torch.device,
                lr: float, epochs: int = EPOCHS,
                patience: int = PATIENCE) -> dict:
    """
    Binary classification training loop with early stopping.
    Returns loss history dict.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history    = {"train_loss": [], "val_loss": []}
    best_loss  = float("inf")
    counter    = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                val_loss += criterion(model(inputs), labels).item()

        avg_train = total_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            counter   = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → Model saved (val_loss={best_loss:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    return history
