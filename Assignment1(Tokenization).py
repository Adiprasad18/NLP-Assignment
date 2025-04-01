import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

text = "Hello, how are you?"

# Tokenize the statement
tokens = word_tokenize(text)
tokens = [word for word in tokens if word.isalnum()]
print(tokens)  # Expected Output: ['Hello', 'how', 'are', 'you']
