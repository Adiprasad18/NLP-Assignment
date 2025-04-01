import nltk
import re
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Input text
text = "This is a sample text. It contains punctuation and stopwords."

# Convert to lowercase
text = text.lower()

# Remove punctuation
text = re.sub(r'[^\w\s]', '', text)

# Tokenize words
words = text.split()

# Remove stopwords
filtered_words = [word for word in words if word not in stopwords.words('english')]

# Join processed words
processed_text = ' '.join(filtered_words)

print(processed_text)  # Expected Output: "sample text contains"
