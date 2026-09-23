import pandas as pd
import numpy as np
import sys


def get_percentile(sorted_data, count, percentile):

    if count == 0:
        return np.nan
    index = percentile * (count - 1)
    lower = int(index)
    upper = lower + 1
    weight = index - lower
    
    if upper >= count:
        return sorted_data[lower]
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

def skewness(clean_data, mean, std, m):
    data = clean_data.to_numpy()
    if m < 3:
        return np.nan
    
    if std == 0:
        return 0.0
    
    factor = m /((m - 1) * (m - 2))
    skew = factor * np.sum(((data - mean) / std) ** 3)

    return skew
    
def kurtosis(clean_data, mean, std, m):
    data = clean_data.to_numpy()
    if m < 4:
        return np.nan

    if std == 0:
        return 0.0

    return (1 / m) * np.sum(((data - mean) / std) ** 4) - 3



def main(path):

    try:
        df = pd.read_csv(path)
        df.set_index("Index", inplace=True)
        numeric_df = df.select_dtypes(include=np.number)
        desc = {}
        for col, series in numeric_df.items():

            clean_data = series.dropna()
            sort = sorted(clean_data)
            count = len(clean_data)
            mean = clean_data.sum() / count if count > 0 else np.nan
            std = (sum((x - mean) ** 2 for x in clean_data) / (count - 1)) ** 0.5
            min = sort[0]
            q1 = get_percentile(sort, count, 0.25)
            median = get_percentile(sort, count, 0.50)
            q3 = get_percentile(sort, count, 0.75)
            max = sort[-1]
            skew = skewness(clean_data, mean, std, count)
            kurt = kurtosis(clean_data, mean, std, count)
            range = max - min

            desc[col] = [count, mean, std, min, q1, median, q3, max, skew, kurt, range]

        result_df = pd.DataFrame(desc, index=['Count', 'Mean', \
                                          'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurt', 'range'])
        print(f"1: {result_df}")
        print(f"2: {df.describe()}")
        print(df.kurtosis(numeric_only=True))
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(-1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: should have a dataset as parameter")
        sys.exit(-1)
    main(sys.argv[1])
