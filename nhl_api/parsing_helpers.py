import numpy as np
from geopy.geocoders import Nominatim
from geopy.adapters import AdapterHTTPError


def convert_height_to_cm(height):
    """ Converts a height in ft and inches to cm.

    The height must be a string of the format 'ft\'inches"'

    Parameters
        height: string = height to be converted

    Returns
        height_cm: float = height in cm
    """

    ft_in = height.split('\'')
    height_cm = 2.54 * (12 * int(ft_in[0]) + int(ft_in[1][:-1]))
    return height_cm


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


def get_venue_coords(city):
    """ Uses a geolocation library to get home team city coordinates.

    TODO: not sure if this will work with the arena name or just the city
    Refs:
    https://anaconda.org/conda-forge/geopy
    https://medium.com/analytics-vidhya/how-to-generate-lat-and-long-coordinates-of-city-without-using-apis-25ebabcaf1d5
    https://towardsdatascience.com/geocode-with-python-161ec1e62b89
    https://nominatim.org/release-docs/latest/api/Search/

    Parameters
        city: str = name of city where the home arena is located

    Returns
        loc.latitude: float = latitude of the citiy
        loc.longitude: float = longitude of the city
    """

    # Initialize the locator
    geolocator = Nominatim(user_agent='myGeocoder')

    # Set the country
    country = 'usa'
    canadian_cities = ['Edmonton', 'Calgary', 'Vancouver', 'Winnipeg',
                       'Toronto', 'Montreal', 'Ottawa']
    if city in canadian_cities:
        country = 'canada'

    # Query the geolocator to find the location
    try:
        loc = geolocator.geocode(f'{city}, {country}')
        return loc.latitude, loc.longitude
    except AdapterHTTPError:
        print('Limit apparently reached')
        return None, None

    # loc = geolocator.geocode(f'{city}, {country}')
    # return loc.latitude, loc.longitude

