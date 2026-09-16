import json
import sys
import numpy as np
import pandas as pd


def sigmoid(z):
        return 1/(1 + np.exp(-z))

def imput_standardize(X, means, stds):
    X = X.fillna(means)
    
    X = (X - means) / stds
  
    return X

def predict(weights, means, stds, features, path):

    try:
    
        df = pd.read_csv(path)
        indices = df["Index"]
        df = df[features]
        X = imput_standardize(df, means, stds)
        X.insert(loc=0, column="bias", value=1)

        houses = list(weights.keys())
        proba = {}

        for house in houses:
            h = sigmoid(np.dot(X, weights[house]))
            proba[house] = h

        proba_df = pd.DataFrame(proba, columns=houses, index=indices)
        # best_house_indices = np.argmax(proba_df, axis=1)
        # print(best_house_indices)
        # predicted_houses = [houses[idx] for idx in best_house_indices]
        # print(predicted_houses)

        predicted_houses = proba_df.idxmax(axis=1)

        submission = pd.DataFrame(
            {"Index": indices, "Hogwarts House": predicted_houses}
        )
        submission.to_csv("houses.csv", index=False)


    except Exception as e:
        print(f"Error: {e}")


def load(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            features = data.get("features", None)
            means = pd.Series(data.get("means", np.zeros(10)), index=features)
            stds = pd.Series(data.get("stds", np.zeros(10)), index=features)
            return data.get("weights", np.zeros(11)), means, stds, features
    except FileNotFoundError as e:
        print(f"Error: {e}. Default value: (0, 0...).")
        return np.zeros(11), np.zeros(10), np.zeros(10), None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: please give a dataset and the weights file")
        sys.exit(-1)
    weights, means, stds, features = load(sys.argv[2])
    predict(weights, means, stds, features, sys.argv[1])