import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x is the input sequence, y is the target (shifted by 1)
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    batch_size = 32
    block_size = 64
    max_iters = 1000
    eval_interval = 100
    learning_rate = 1e-3
    vocab_size = 512 # small vocab for nano model
    
    # 1. Load/Prepare Data
    # For demonstration, we'll use a small snippet of text if no data exists
    data_path = 'input.txt'
    if not os.path.exists(data_path):
        with open(data_path, 'w') as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 100)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 2. Tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    print("Training tokenizer...")
    tokenizer.train(text)
    tokenizer.save('tokenizer.json')
    
    train_data = tokenizer.encode(text)
    print(f"Dataset has {len(train_data)} tokens")
    
    dataset = TextDataset(train_data, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    model = GPT(config).to(device)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 5. Training Loop
    model.train()
    iter_num = 0
    start_time = time.time()
    
    data_iter = iter(loader)
    
    for i in range(max_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if i % eval_interval == 0:
            print(f"iter {i}: loss {loss.item():.4f}")
            
    # Save the model
    torch.save(model.state_dict(), 'model.pt')
    print(f"Training complete. Model saved to model.pt")

if __name__ == "__main__":
    train()