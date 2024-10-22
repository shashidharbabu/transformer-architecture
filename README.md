# Transformer Model in PyTorch

This project involves implementing the Transformer architecture using PyTorch. The Transformer is a model commonly used for machine translation, text generation, and other tasks.


## Architecture

![Screenshot 2024-10-22 at 15 50 24](https://github.com/user-attachments/assets/09084b0b-be0c-4794-b38e-9ee1ae060ffe)


## Features
- **Self-Attention**: Mechanism to allow the model to focus on different parts of the input sequence.
- **Multi-Head Attention**: Multiple attention heads process different parts of the input simultaneously.
- **Encoder-Decoder Architecture**: The model consists of an encoder that processes the input sequence and a decoder that generates the output sequence.

## How to Use

### Example Code

```python
import torch
from transformer import Transformer

# Example input and target tensors
x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to('cpu')
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to('cpu')

# Model configuration
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to('cpu')

# Forward pass
output = model(x, trg[:, :-1])
print(output.shape)
```

### Steps to Run:
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   ```
2. **Install PyTorch** (if you don't have it already):
   ```bash
   pip3 install torch torchvision torchaudioo
   ```
3. **Run the example**:
   ```bash
   python main.py
   ```

## Model Overview
The model consists of an encoder and decoder, each built with layers of self-attention and feed-forward networks. It processes sequences to generate translations or other sequence-based outputs.

## Resources:
Paper: https://arxiv.org/abs/1706.03762

