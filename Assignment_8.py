import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
from collections import Counter

nltk.download('punkt')

# Sample text dataset
text = "Machine learning allows computers to learn from data."

# Tokenization & Preprocessing
words = nltk.word_tokenize(text.lower())
words = [re.sub(r'[^a-z]', '', word) for word in words if word.isalpha()]

# Build Vocabulary
word_counts = Counter(words)
vocab = {word: i for i, word in enumerate(word_counts.keys())}
vocab_size = len(vocab)

# Convert words to indices
def word_to_index(word):
    return torch.tensor([vocab[word]], dtype=torch.long)

# Generate Skip-gram Data
def generate_skipgram_data(words, window_size=2):
    pairs = []
    for i, target in enumerate(words):
        context = words[max(0, i - window_size): i] + words[i + 1: i + window_size + 1]
        for ctx in context:
            pairs.append((target, ctx))
    return pairs

skipgram_data = generate_skipgram_data(words)

# Skip-gram Model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embed = self.embeddings(target)
        output = self.linear(embed)
        return output

# Train Model
embedding_dim = 10
model = SkipGram(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for target, context in skipgram_data:
        target_idx = word_to_index(target)
        context_idx = word_to_index(context)

        optimizer.zero_grad()
        output = model(target_idx)
        loss = criterion(output, context_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
