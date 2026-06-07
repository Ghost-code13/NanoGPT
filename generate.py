import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None, device='cpu'):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # crop idx to the maximum block size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
    return tokenizer.decode(idx[0].tolist())

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    tokenizer = BPETokenizer.load('tokenizer.json')
    
    # Load model
    config = GPTConfig(
        vocab_size=512, # must match training
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128
    )
    model = GPT(config).to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    
    # Generate
    prompt = "First Citizen:"
    print(f"Prompt: {prompt}")
    print("-" * 30)
    generated_text = generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=10, device=device)
    print(generated_text)

if __name__ == "__main__":
    main()