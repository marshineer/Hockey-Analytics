import numpy as np
import pandas as pd


# Modelling TODOs
# TODO: Should the rink be divided into zones, or left as a continuous variable?

def calc_travel_dist(city1, city2):
    """ Calculates the distance (km) between two cities.

    The city locations are given in latitude and longitude. This function is
    used to approximate the travel distances between cities for NHL teams. The
    city center is used as the location, rather than the precise location of
    the airport or arena, as I doubt the additional precision will significantly
    affect the model outcome.

    The function uses the Haversine formula
    http://www.movable-type.co.uk/scripts/latlong.html

    Parameters
        city1: tuple = (latitude, longitude) coordinates of the first city
        city2: tuple = (latitude, longitude) coordinates of the second city

    Returns
        dist: int = distance between the two cities (km)
    """

    # Convert the coordinates to radians
    phi1_rad = np.radians(city1[0])
    lambda1_rad = np.radians(city1[1])
    phi2_rad = np.radians(city2[0])
    lambda2_rad = np.radians(city2[1])

    # Calculate the distance using the Haversine formula
    R = 6371  # Earth's radius (km)
    phi_diff = phi2_rad - phi1_rad
    lambda_diff = lambda2_rad - lambda1_rad
    a1 = np.sin(phi_diff / 2)**2
    a2 = np.cos(phi1_rad) * np.cos(phi2_rad) * np.sin(lambda_diff / 2)**2
    c = np.arctan2(np.sqrt(a1 + a2), np.sqrt(1 - (a1 + a2)))
    dist = R * c

    return dist


def binary_encoder(data, col_ls):
    """ Encodes numericized categorical data as a set of binary columns.

    Categorical data needs to be encoded such that it is usable by ML algorithms.
    However, large numbers of categories in one-hot encoding leads to large
    feature dimensions. Therefore, this function uses binary encoding to reduce
    the space required to encode these features. Binary encoding avoids the issue
    of collisions, which occurs with other methods such as hash encoding.

    The length of the binary encoding required is calculated based on the number
    of unique categories for each column. The encoded columns are concatenated
    at the beginning of the data frame.

    This function requires that the categorical data has already been converted
    to be represented by a range of ints. This will be updated in later versions
    of the function.

    Paramters
        data: DataFrame = all data
        col_ls: list = list of categorical column names to be encoded

    Returns
        enc_data: DataFrame = categorical data encoded columns
        n_enc_col: int = total number of encoded columns in output
    """

    n_enc_col = 0
    for i, col in enumerate(col_ls):
        # Find the number of unique categories
        n_cats = len(data[col].unique())

        # Determine the length of the binary encoding
        n_dig = len(format(n_cats - 1, 'b'))
        n_enc_col += n_dig

        # Calculate the encoding for each category
        bin_enc = np.zeros((len(data), n_dig), dtype=int)
        for n in range(n_cats):
            inds = data.loc[data[col] == n].index
            bin_val = [int(x) for x in format(n, f'0{n_dig}b')]
            bin_enc[inds, :] = bin_val

        # Assign new column names
        col_names = []
        for d in range(n_dig):
            col_names.append(f'{col}{d}')

        # Concatenate the encoded columns
        if i == 0:
            enc_data = pd.DataFrame(bin_enc, columns=col_names)
        else:
            enc_data = pd.concat([pd.DataFrame(bin_enc, columns=col_names),
                                  enc_data], axis=1)

    return enc_data, n_enc_col
