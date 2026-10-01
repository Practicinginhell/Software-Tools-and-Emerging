## Refactored Code

import pandas as pd


def load_data(source: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(source)


def calculate_average(df: pd.DataFrame, column: str) -> float:
    """Calculate the average of a column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    return df[column].mean()


def find_maximum(df: pd.DataFrame, column: str) -> float:
    """Find the maximum value in a column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    return df[column].max()


def filter_species(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Return rows for the specified species."""
    if "species" not in df.columns:
        raise ValueError("Column 'species' does not exist.")

    return df[df["species"] == species]


def main():
    source = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    df = load_data(source)

    average = calculate_average(df, "sepal_length")
    maximum = find_maximum(df, "petal_width")
    setosa = filter_species(df, "setosa")

    print("Average sepal length:", average)
    print("Max petal width:", maximum)
    print(setosa.head())


if __name__ == "__main__":
    main()
