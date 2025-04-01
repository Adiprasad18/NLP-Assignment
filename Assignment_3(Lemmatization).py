import nltk
from nltk.stem import WordNetLemmatizer

# Download WordNet data
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Input word
word = "running"

# Perform lemmatization
lemma = lemmatizer.lemmatize(word, pos='v')  # 'v' for verb form

print(lemma)  # Expected Output: 'run'
