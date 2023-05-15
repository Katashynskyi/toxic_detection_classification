import json

import spacy

nlp = spacy.load("en_core_web_md")

# Example comments data
comments = ["This is great!", "I really enjoyed this movie.", "Terrible service."]

# Convert comments to Doc objects
docs = list(nlp.pipe(comments))

# Train word vectors
nlp.vocab.vectors.name = "my_custom_embeddings"

nlp.vocab.train_vectors(docs)

nlp = spacy.blank("en")
doc = nlp("Czech Republic may help Slovakia protect its airspace")

# Import the PhraseMatcher and initialize it
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)

# Create pattern Doc objects and add them to the matcher
# This is the faster version of: [nlp(country) for country in COUNTRIES]
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", patterns)

# Call the matcher on the test document and print the result
matches = matcher(doc)
print([doc[start:end] for match_id, start, end in matches])
