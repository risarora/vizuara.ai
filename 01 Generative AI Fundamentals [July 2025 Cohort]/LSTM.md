https://www.youtube.com/watch?v=WCUNPb-5EYI&ab_channel=BrandonRohrer
https://www.youtube.com/watch?v=YCzL96nL7j0&ab_channel=StatQuestwithJoshStarmer

These are results for increase hidden layers in LSTM
Search instead for uncrese hidden layers in LSTM

AI Overview
Understanding of LSTM Networks - GeeksforGeeks
To increase hidden layers in an LSTM model, stack multiple LSTM layers in your model, making sure to set return_sequences=True for all but the final LSTM layer to preserve the temporal dimension for the subsequent layers. This increases the depth of the network, allowing it to learn more complex temporal patterns, but it also increases training time and the risk of overfitting, so you should use a validation set to tune the number of layers.

```python
# Example in Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=44000, output_dim=32)) # Input layer
# First LSTM layer: return_sequences=True because it feeds into another LSTM layer
model.add(LSTM(32, return_sequences=True))
# Second LSTM layer: return_sequences=True because it feeds into another LSTM layer
model.add(LSTM(32, return_sequences=True))
# Third LSTM layer: return_sequences=False (default) as it's the last LSTM layer before the output
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid')) # Output layer
```

Key Considerations

- `return_sequences=True`:

  This is crucial for intermediate LSTM layers to ensure they pass their entire output sequence (rather than just the last timestep's output) to the next layer.

- **Complexity vs. Overfitting**:

  More hidden layers can improve the model's capacity to learn intricate patterns, but too many layers can lead to overfitting (the model learns the training data too well but performs poorly on new data).

- **Training Time**:

  A deeper network with more layers requires more computational resources and takes longer to train.

- **Hyperparameter Tuning**:

  Experimentation is key. Use a validation set to find the optimal number of hidden layers and units for your specific task and data. You might start with a single layer and progressively add more, observing the performance on the validation set to guide your decisions.

# 2. PyTorch Dataset and DataLoader

class IMDBDataset(Dataset):
def **init**(self, sequences, labels):
self.sequences = sequences
self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

train_dataset = IMDBDataset(train_sequences, train_labels)
test_dataset = IMDBDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# 3. Define LSTM Model

class SentimentLSTM(nn.Module):
def **init**(self, vocab_size, embed_dim, hidden_size, num_layers):
super().**init**()
self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
self.dropout = nn.Dropout(0.5)
self.fc = nn.Linear(hidden_size, 1)
self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentLSTM(vocab_size=len(vocab), embed_dim=128, hidden_size=64, num_layers=1).to(device)

# 4. Training setup

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. Evaluation (basic accuracy)

model.eval()
correct, total = 0, 0
with torch.no_grad():
for sequences, labels in test_loader:
sequences, labels = sequences.to(device), labels.to(device)
outputs = model(sequences)
preds = (outputs > 0.5).float()
correct += (preds == labels).sum().item()
total += labels.size(0)
print(f"Test Accuracy: {correct / total:.4f}")
