from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset
documents = ["This is a sample document.", "Another document with different content."]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Compute TF-IDF Matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Display TF-IDF Matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Display Feature Names
print("Feature Names:", vectorizer.get_feature_names_out())
