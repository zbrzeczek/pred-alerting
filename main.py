from DanceDataset import DanceDataset
import os
from lstm import LstmModel
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.float().to(device)
        y = y.long().to(device)

        optimizer.zero_grad()
        outputs = model(X) # co nasz model wypluwa teraz

        loss = criterion(outputs, y)
        loss.backward() # back propagation czy jakos tak, poprawa na podstawie funcji straty
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim = 1)
        total += y.size(0)
        correct += (preds == y).sum().item()

    avg_loss = total_loss/ len(loader)
    acc = correct / total
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.float().to(device)
        y = y.long().to(device)

        outputs = model(X) # co nasz model wypluwa teraz
        loss = criterion(outputs, y)

        total_loss += loss.item()
        preds = outputs.argmax(dim = 1)
        total += y.size(0)
        correct += (preds == y).sum().item()

    avg_loss = total_loss/ len(loader)
    acc = correct / total
    return avg_loss, acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # zbiera juz przerobione i oznaczone dane do trenowania i val
    train_dataset = DanceDataset("./processed/train", seq_len=60)
    val_dataset   = DanceDataset("./processed/val", seq_len=60)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,
        shuffle=False
    ) 

    X_batch, y_batch = next(iter(train_loader))
    print("Batch shape:", X_batch.shape)
    print("Label batch:", y_batch)

    model = DanceLSTM(
        input_size=36,
        hidden_size=128,
        num_classes=len(LABEL_MAP)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 30

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
        )

    # Save model
    torch.save(model.state_dict(), "dance_lstm.pth")
    print("Model saved to dance_lstm.pth")