import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Assignment 4: Semantic Similarity using Cosine Similarity
# Sentences for comparison
sentence1 = "This is a sample sentence."
sentence2 = "This sentence is just a sample."

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Compute TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])

# Compute Cosine Similarity
similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

print(f"Similarity Score: {round(similarity_score, 2)}")  # Expected Output: ~0.8
