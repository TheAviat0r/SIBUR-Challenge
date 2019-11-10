import pandas as pd

from sklearn.datasets import load_iris

DATA_PATH = "data/dummy_dataset.pkl"

data = load_iris()

X = data["data"]
y = data["target"]

columns = ["x1", "x2", "x3", "x4"]

df = pd.DataFrame(X, columns=columns)
df["uid"] = range(X.shape[0])

df.to_pickle(DATA_PATH)
