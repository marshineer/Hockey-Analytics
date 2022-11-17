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


def get_venue_coords(city):
    """ Uses a geolocation library to get home team city coordinates.

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
