import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "model.pt"
TRAIN_TEXT_PATH = "train.txt"   # put your training text here
SAVE_PATH = "model_continued.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 512
BLOCK_SIZE = 64
N_EMBD = 128
N_LAYER = 4
N_HEAD = 4
BATCH_SIZE = 32
LR = 3e-4
STEPS = 1000
EVAL_EVERY = 100

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(VOCAB_SIZE, N_EMBD),
            wpe=nn.Embedding(BLOCK_SIZE, N_EMBD),
            h=nn.ModuleList([Block() for _ in range(N_LAYER)]),
            ln_f=nn.LayerNorm(N_EMBD),
        ))
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def encode(text):
    return list(text.encode("utf-8"))

def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

with open(TRAIN_TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
if len(data) < BLOCK_SIZE + 2:
    raise ValueError("train.txt is too small. Add more text.")

model = GPT().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

model.train()
for step in range(1, STEPS + 1):
    xb, yb = get_batch(data)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == 1:
        print(f"step {step}/{STEPS} | loss {loss.item():.4f}")

# Save continued model weights
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved continued model to {SAVE_PATH}")
