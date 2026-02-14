#!/usr/bin/env python
# coding: utf-8
"""
Script for training an MLP-based sentiment classifier on the financial_phrasebank dataset.
Uses FastText word embeddings (mean-pooled) as input features.
"""

# ========== Imports ==========
import numpy as np
import pandas as pd
import datasets
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ========== Reproducibility ==========
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print("\n========== Loading Dataset ==========")
# ========== Load Dataset ==========
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")

# ========== Load FastText Model ==========
print("\n========== Loading FastText Model ==========")
fasttext_model = KeyedVectors.load('../Assignment_2/fasttext-wiki-news-subwords-300.model')
emb_dim = fasttext_model.vector_size
print(f"FastText model loaded. Embedding dimension: {emb_dim}")

# ========== Compute Mean-Pooled Embeddings ==========
print("\n========== Computing Mean-Pooled Embeddings ==========")

def mean_pool_embedding(sentence, model, emb_dim):
    """Compute mean of FastText word vectors for a sentence. Skip OOV words."""
    tokens = sentence.lower().split()
    vectors = []
    for token in tokens:
        if token in model:
            vectors.append(model[token])
    if len(vectors) == 0:
        return np.zeros(emb_dim)
    return np.mean(vectors, axis=0)

X = np.array([mean_pool_embedding(sent, fasttext_model, emb_dim) for sent in tqdm(data['sentence'], desc="Embedding sentences")])
y = data['label'].values
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ========== Train/Test Split ==========
print("\n========== Splitting Data ==========")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ========== PyTorch Dataset ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EmbeddingDataset(X_train, y_train)
val_dataset = EmbeddingDataset(X_val, y_val)
test_dataset = EmbeddingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("DataLoaders created.")

# ========== Model Definition ==========
print("\n========== Defining MLP Model ==========")
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

num_classes = len(np.unique(y))
model = MLPClassifier(emb_dim, num_classes)
print(f"Model initialized with input_dim={emb_dim}, num_classes={num_classes}")

# ========== Training Setup ==========
print("\n========== Setting Up Training ==========")
def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

device = get_device()
print(f"Using device: {device}")
os.makedirs("outputs", exist_ok=True)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
counts = np.bincount(y_train)
class_weights = 1. / torch.tensor(counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Training setup complete.")

# ========== Training Loop ==========
print("\n========== Starting Training Loop ==========")
num_epochs = 50
best_val_f1 = 0.0
train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()
    train_loss_history.append(epoch_train_loss)
    train_f1_history.append(train_f1)
    train_acc_history.append(train_acc)
    print(f"Train Loss: {epoch_train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
    # Validation
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()
    val_loss_history.append(epoch_val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)
    print(f"Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
    scheduler.step(val_f1)
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}')
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'outputs/best_mlp_model.pth')
        print(f'>>> Saved new best model (Val F1: {best_val_f1:.4f})')

# ========== Plot Learning Curves ==========
print("\n========== Plotting Learning Curves ==========")
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/mlp_f1_learning_curves.png')
plt.close()
print("Learning curves saved as 'outputs/mlp_f1_learning_curves.png'.")

# Save accuracy plot separately
plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/mlp_accuracy_learning_curve.png')
plt.close()
print("Accuracy curve saved as 'outputs/mlp_accuracy_learning_curve.png'.")

# ========== Test Evaluation ==========
print("\n========== Evaluating on Test Set ==========")
model.load_state_dict(torch.load('outputs/best_mlp_model.pth'))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
print('\n' + '='*50)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print('='*50 + '\n')
class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('outputs/mlp_confusion_matrix.png')
plt.close()
print("Confusion matrix saved as 'outputs/mlp_confusion_matrix.png'.")
print("\nPer-class F1 Scores:")
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro')
    print(f"{name}: {class_f1:.4f}")

print("\n========== Script Complete ==========")
# End of script
