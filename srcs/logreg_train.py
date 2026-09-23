import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=None):
        self.lr = learning_rate
        self.weights = None
        self.loss = []
        self.epochs = epochs
        self.batch_size = batch_size

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        b_size = self.batch_size if self.batch_size is not None else m

        for _ in range(0, self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = y[indices]

            for start in range(0, m, b_size):
                end = start + b_size
                X_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]
                b_len = len(X_batch)

                h = self.sigmoid(np.dot(X_batch, self.weights))
                gradient = (1 / b_len) * np.dot(X_batch.T, (h - y_batch))
                self.weights -= self.lr * gradient

            h_all = self.sigmoid(np.dot(X, self.weights))
            h_all = np.clip(h_all, 1e-15, 1 - 1e-15)
            loss = (-1 / m) * np.sum(y * np.log(h_all) + (1 - y) * np.log(1 - h_all))
            self.loss.append(loss)

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
        all_loss = {}
        for house in houses:
            y = all_y[house]

            model = LogisticRegression(0.1, 1000)           #batch
            # model = LogisticRegression(0.1, 100, 32)      #minibatch
            # model = LogisticRegression(0.01, 100, 1)      #stochastic
            weights = model.fit(X, y)

            res[house]= weights.tolist()
            all_loss[house] = model.loss

        output = {
            "weights": res,
            "features": features.tolist(),
            "means": means.tolist(),
            "stds": stds.tolist()
        }

        write_json(output)

        loss_df = pd.DataFrame(all_loss)
        sns.lineplot(loss_df)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss function")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        raise
        sys.exit(-1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: you should give a dataset as param")
        sys.exit(-1)
    train(sys.argv[1])