import numpy as np


# Modelling TODOs
# TODO: Should the rink be divided into zones, or left as a continuous variable?

# TODO: this is a modelling function
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


