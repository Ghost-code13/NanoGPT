import re
import json
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def get_stats(self, ids):
        counts = Counter()
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i+1])] += 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        # Convert text to bytes
        tokens = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256
        
        ids = list(tokens)
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, 
pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            # Find the pair that occurred first in our merges list
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            tokens = self.merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, path):
        # Convert tuple keys to strings for JSON
        serializable_merges = {f"{p[0]},{p[1]}": v for p, v in self.merges.items()}
        data = {
            "vocab_size": self.vocab_size,
            "merges": serializable_merges
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        # Convert string keys back to tuples
        tokenizer.merges = {tuple(map(int, k.split(','))): v for k, v in data["merges"].items()}
        # Reconstruct vocab
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in sorted(tokenizer.merges.items(), key=lambda x: x[1]):
            tokenizer.vocab[idx] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        return tokenizer 

if __name__ == "__main__":
    # Test the tokenizer
    text = "Building an LLM from scratch is fun! Hello world, hello LLM."
    tokenizer = BPETokenizer(vocab_size=260) # Small vocab for testing
    tokenizer.train(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    assert text == decoded
