from os.path import exists
import csv
from geopy.geocoders import Nominatim


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

    If any exception occurs, return (None, None). This is left intentionally
    broad because this data is non-critical, and it can easily be found later.
    It is not worth handling each individual exception or breaking the program
    for when this operation does not work.

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
    except Exception as e:
        print(e)
        return None, None


def check_dict_exist(fpath, id_key):
    """ Return contents of a file if it exists, empty dict otherwise.

    Check whether a .csv file containing the coach, team or player data exists.
    If it does, pull the data and store it in a dictionary to be added to. If
    the file does not exist, return an empty dictionary.

    Parameters
        fpath: str = relative path to the file
        id_key: str = the unique key used to identify individual entities

    Returns
        : dict = all the coaches, teams, or players in the given file
    """
    try:
        with open(fpath, 'r') as f:
            dict_reader = csv.DictReader(f)
            entity_dict = list(dict_reader)
        return {entity_x[id_key]: entity_x for entity_x in entity_dict}
    except FileNotFoundError:
        return {}


def save_nhl_data(fpath, data_dict, overwrite=False):
    """ Save the data to a .csv file.

    Lists of flat dictionaries are saved to .csv files with keys as headers and
    rows consisting of entries in the list. This requires that each dictionary
    contains the same set of keys.

    If the data is for games (game info, box scores, events, or shifts), append
    to existing files or create new files. Only write a header if creating a new
    file. If the data is a list of coaches, teams or players, overwrite the
    existing file (always write a header).

    Parameters
        fpath: str = relative path to where the file should be saved
        data_dict: dict = dictionary of data to be saved
        overwrite: bool = indicates whether the data should be overwritten
    """
    field_names = data_dict[0].keys()
    new_file = not exists(fpath)
    f_mode = 'w' if overwrite else 'a'
    with open(fpath, f_mode) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if overwrite or new_file:
            writer.writeheader()
        writer.writerows(data_dict)
        csvfile.close()
