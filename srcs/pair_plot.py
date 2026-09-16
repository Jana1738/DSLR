import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys


def plot(path):
    try:
        df = pd.read_csv(path)
        df.set_index("Index", inplace=True)

        course_cols = df.select_dtypes(include="number").columns
        data_to_plot = df[["Hogwarts House"] + list(course_cols)].dropna()

        g = sns.pairplot(
            data_to_plot,
            hue="Hogwarts House",
            palette={
                "Gryffindor": "red",
                "Hufflepuff": "gold",
                "Ravenclaw": "blue",
                "Slytherin": "green",
            },
            markers="o",
            plot_kws={"alpha": 0.5, "s": 15},
            diag_kind="kde",
        )

        g.figure.suptitle("Pair Plot: Hogwarts Courses Comparison", y=1.02)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please give a dataset")
        sys.exit(-1)
    plot(sys.argv[1])