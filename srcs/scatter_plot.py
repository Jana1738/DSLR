import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def plot(df, course_1, course_2):
    plt.scatter(df[course_1], df[course_2])
    plt.title(f"Correlation between {course_1} and {course_2}")
    plt.xlabel(course_1)
    plt.ylabel(course_2)
    plt.legend
    plt.show()


def find_corr(path):
    try:
        df = pd.read_csv(path)
        df.set_index("Index", inplace=True)

        courses = df.select_dtypes(np.number)
        corr_matrix = courses.corr()
        print(corr_matrix)
        corr_pairs = corr_matrix.unstack().reset_index()
        corr_pairs.columns = ["Course_1", "Course_2", "Correlation"]
        corr_pairs = corr_pairs[corr_pairs["Course_1"] != corr_pairs["Course_2"]]

        corr_pairs["Abs_Corr"] = corr_pairs["Correlation"].abs()
        sorted_pairs = corr_pairs.sort_values(by="Abs_Corr", ascending=False)

        print(sorted_pairs.iloc[::2].head(3))

        plot(df, "Astronomy", "Defense Against the Dark Arts")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(-1)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please give a dataset")
        sys.exit(-1)
    find_corr(sys.argv[1])
