import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Input sentence
sentence = "John Smith works at Google."

# Tokenize and POS tagging
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)

# Perform Named Entity Recognition (NER)
ner_tree = ne_chunk(pos_tags)

# Extract named entities
named_entities = []
for subtree in ner_tree:
    if hasattr(subtree, 'label'):
        entity_name = " ".join([token for token, pos in subtree.leaves()])
        entity_type = subtree.label()
        named_entities.append((entity_name, entity_type))

print(named_entities)  # Expected Output: [('John Smith', 'PERSON'), ('Google', 'ORGANIZATION')]
