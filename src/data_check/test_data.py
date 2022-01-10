import pandas as pd
import numpy as np
import scipy.stats
import argparse


def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold

def test_row_count(data: pd.DataFrame):
    """
    Test of proper data rows range
    """
    assert 15000 < data.shape[0] < 1000000

def test_price_range(data: pd.DataFrame, min_price, max_price):
    """Test of proper boundaries of price values"""
    assert data['price'].between(min_price, max_price).all()

def go(args):
    '''go func of whole step'''
    df = pd.read_csv(args.input_artifact)
    ref_df = pd.read_csv(args.ref_artifact)
    test_column_names(df)
    test_neighborhood_names(df)
    test_proper_boundaries(df)
    test_similar_neigh_distrib(df, ref_df, args.kl_threshold)
    test_row_count(df)
    test_price_range(df, args.min_price, args.max_price)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="The data verification tests")


    parser.add_argument(
        "--csv", 
        type=str,
        help="input artifact for data check step",
        required=True
    )

    parser.add_argument(
        "--ref", 
        type=str,
        help="refernece artifact for data check step",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="the minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="the maximum price to consider",
        required=True
    )

    parser.add_argument(
        "--kl_threshold", 
        type=float,
        help="the maximum price to consider",
        required=True
    )


    args = parser.parse_args()

    go(args)
