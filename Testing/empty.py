import spacy
# Load a larger pipeline with vectors
nlp = spacy.load("en_core_web_md")

doc = nlp("I have a banana")

# Access the vector via the token.vector attribute
print(doc[3].vector)

# Print the vector representation of each token
for token in doc:
    print(token.text)#, token.vector)
    print(token.shape)
    print(len(token))