#!/usr/bin/env python3
"""
Pneumonia Classification from Chest X-Ray Images
Uses transfer learning with ResNet18 for 3-class classification:
- BACTERIAL_PNEUMONIA
- NORMAL
- VIRAL_PNEUMONIA
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "pneumonia_model.pth")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 3
IMAGE_SIZE = 224

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_data_transforms():
    """Define data augmentation and normalization transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def load_data():
    """Load training and test datasets."""
    train_transform, test_transform = get_data_transforms()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader, train_dataset.classes


def create_model():
    """Create a ResNet18 model with pretrained weights."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, NUM_CLASSES)
    )

    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion):
    """Evaluate the model on test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)

    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to confusion_matrix.png")


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history saved to training_history.png")


def main():
    print("=" * 60)
    print("Pneumonia Classification Training")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_loader, test_loader, classes = load_data()

    # Create model
    print("\nCreating model...")
    model = create_model()
    print(f"Model: ResNet18 (pretrained, fine-tuned)")

    # Loss function with class weights to handle imbalance
    class_counts = [7560, 4044, 4014]  # BACTERIAL, NORMAL, VIRAL
    weights = torch.FloatTensor([1.0/c for c in class_counts])
    weights = weights / weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on test set
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'classes': classes
            }, MODEL_SAVE_PATH)
            print(f"Model saved (new best accuracy: {best_acc:.2f}%)")

        scheduler.step()

    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

    # Final evaluation with best model
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_acc, predictions, labels = evaluate(model, test_loader, criterion)

    print(f"\nTest Accuracy: {final_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=classes))

    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, classes)

    print("\nTraining complete!")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
