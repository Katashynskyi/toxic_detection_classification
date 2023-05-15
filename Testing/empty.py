import numpy as np
import spacy
from guppy import hpy

h = hpy()
from scipy.sparse import csr_matrix, hstack

nlp = spacy.load("en_core_web_sm")
a = ["some text 1", "some text 2", "some text 3"]
# Doc = np.array([nlp(text).vector for text in a])
# Doc = Doc.reshape((len(a), -1))
# Doc=csr_matrix(Doc)
# print(Doc)
# print(Doc.shape)

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#
# newarr = arr.reshape(4, 3)
#
# print(newarr)


Doc = np.array([nlp(text).vector for text in a])
# print(Doc)
print(type(Doc))
Doc = [nlp(text).vector for text in a]
# print(Doc)
print(type(Doc))
print(h.heap())
