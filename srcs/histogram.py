import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def plot_histo(path):
    try:
        df = pd.read_csv(path)
        df.set_index("Index", inplace=True)

        courses = df.select_dtypes(include=np.number).columns
        n_courses = len(courses)

        nb_col = 4
        if n_courses % 4 == 0:
            nb_row = int(n_courses / nb_col)
        else:
            nb_row = int((n_courses / nb_col) + 1)
        fig, axes = plt.subplots(nb_row, nb_col,figsize=(16, 12))
        axes = axes.flatten()

        grouped = df.groupby("Hogwarts House")

        for i, course in enumerate(courses):
            ax = axes[i]
            for house_name, group in grouped:
                ax.hist(group[course].dropna(), alpha=0.5, label=house_name, bins=25)
            ax.set_title(course, fontsize=10)
            ax.tick_params(labelsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

        plt.tight_layout()

        df_z = (df[courses] - df[courses].mean()) / df[courses].std()
        df_z['Hogwarts House'] = df['Hogwarts House']
        homogeneity = df_z.groupby('Hogwarts House').mean().std().sort_values()
        print(homogeneity)


        plt.show()
 
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(-1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please give a dataset")
        sys.exit(-1)
    plot_histo(sys.argv[1])
