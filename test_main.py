"""
Test goes here

"""
import pandas as pd
import os


def test_dataset():
    """
    Test that the dataset is available
    """
    assert os.path.exists("data/wwc_matches_1.csv")
    df = pd.read_csv("data/wwc_matches_2.csv")
    assert df.shape[0] > 0
    assert df.shape[1] > 0


if __name__ == "__main__":
    test_dataset()
