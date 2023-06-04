import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
y = np.array([0, 0, 1, 1])

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)
for i in skf.split(X, y):
    print(i)

    # print(type(i))
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}")
#     print(f"  Test:  index={test_index}")
