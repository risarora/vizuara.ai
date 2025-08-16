# Lecture 5 : RNN in pytorch

https://zoom.us/clips/share/7BfjyGIRSCut3-SoLVTz_w

### Description

**The speaker discusses the process of training a neural network for English to French translation using an encoder-decoder architecture. They explain the concept of memorization and how it can be used to train a model. The speaker also discusses the use of PyTorch for building the neural network and how it can be used to simulate an OR gate. They explain the process of text pre-processing and how it can be used to clean the dataset. The speaker also discusses the concept of batching and how it can be used to reduce the computational cost of training the model. They explain the process of training the model and how the weights are adjusted during the backpropagation step. The speaker also discusses the concept of hyperparameter optimization and how it can be used to find the best parameters for the model. Finally, the speaker shows the output of the model and discusses how it can be improved by adjusting the hyperparameters.**

### Encoder-Decoder Architecture in Translation and Summarization - 00:04:10

**The encoder-decoder architecture, which is particularly useful in translation and summarization tasks.**

- Simple RNN cell structure
- Encoder decoder architecture
- Translation Use Case
  - encoder
  - decoder

### Coding Encoder-Decoder Architecture in PyTorch for Translation - 00:10:14

**The process of coding an encoder-decoder architecture in PyTorch for English to French translation. I introduced the concept of PyTorch and its usage in creating a small neural network.**

### Creating a Simple Neural Network With PyTorch - 00:17:59

**How to create a simple neural network using PyTorch. I demonstrated how to define a model with two input neurons, a hidden layer with three neurons, and an output layer with one neuron.**

```python
import os, io, math, random, re, zipfile, urllib.request
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

```python
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

```python
X_train = torch.tensor([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]], dtype=torch.float32)

y_train = torch.tensor([[0],
                        [1],
                        [1],
                        [1]], dtype=torch.float32)

```

```python

X_train


```

- `Output`

```python
tensor([[0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]])
```

```python
class ORGateNet(nn.Module):
    def __init__(self):
        super(ORGateNet, self).__init__()
        # Simple network: 2 inputs -> 3 hidden units -> 1 output
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x)) # 1 dimensional tensor
        return x
```

```python
model = ORGateNet()
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.SGD(model.parameters(), lr=5.0) # lr 10^-3
```

```python

```

### English-French Pairs for Text Pre-Processing - 00:34:09

**The use of the English-French pairs from the official PyTorch tutorial trade for text pre-processing.**

### Sorting Words by Frequency and Assigning Order - 00:40:28

**The process of sorting words by frequency of occurrence and assigning them in a specific order.**

### Splitting Dataset Into Training and Validation Pairs - 00:46:01

**The process of splitting a dataset into training pairs and validation pairs. I explained how to call the defined functions, such as encode and padding, and convert them into tensors.**

### Multiplying Hidden State, Initializing Tensors, and RNN Cell - 00:53:45

**The process of multiplying with the hidden state and initializing tensors. I also discussed the forward pass in the RNN cell and the encoder-decoder model.**

### Reshaping Ground Truth Data and Output Data - 01:06:27

**How we use ground truth data and model inputs in batches. I demonstrated how we reshape both the ground data and output data to match their dimensions.**

### Training a Language Translation Model - 01:11:02

**The process of training a language translation model.**

### IDS2Tix Function in Translation Process - 01:18:07

**The IDS2Tix function and its role in the translation process. I explained that this function is an auxiliary function that doesn't require much attention.**
