import pandas as pd
import seaborn as sn
import numpy as np
import sys
import json


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.ite = iterations
        self.weights = None

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        for _ in range(0, self.ite):
            h = self.sigmoid(np.dot(X, self.weights))
            gradient = (1 / m) * np.dot(X.T, (h - y))
            self.weights -= self.lr * gradient
        
        return self.weights


def write_json(data):
    try:
        with open("model.json", "w") as f:
            json.dump(data, f, indent=4)
        print(f"Model saved in model.json")
    except Exception as e:
        print(f"Error: {e}")



def imput_and_standardize(X: pd.DataFrame):
    means = X.mean()
    X = X.fillna(means)
    
    stds = X.std()
    X = (X - means) / stds
  
    return X, means, stds


def train(path: str):
    try:
        df = pd.read_csv(path)
        df.set_index("Index", inplace=True)

        X = df.select_dtypes(np.number)
        X.drop(columns=["Arithmancy", "Care of Magical Creatures", "Astronomy"], inplace=True)
        features = X.columns
        X, means, stds = imput_and_standardize(X)
        X.insert(loc=0, column="bias", value=1)
        X = X.to_numpy()
        
        houses = df["Hogwarts House"].dropna().unique()
        all_y = {house: (df["Hogwarts House"] == house).astype(int).to_numpy() for house in houses}

        res = {}
        for house in houses:
            y = all_y[house]

            model = LogisticRegression(0.1, 1000)
            weights = model.fit(X, y)

            res[house]= weights.tolist()

        output = {
            "weights": res,
            "features": features.tolist(),
            "means": means.tolist(),
            "stds": stds.tolist()
        }

        write_json(output)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(-1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: you should give a dataset as param")
        sys.exit(-1)
    train(sys.argv[1])