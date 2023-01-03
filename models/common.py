import numpy as np
import psycopg2 as psy
import pandas as pd


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


def create_db_connection(db, user, host, port, password=None):
    """ Returns a database connection for executing sql commands.

    Parameters
        db: str = database name
        user: str = username
        host: str = host name (IP)
        port: str = port number
        password: str = password (optional)

    Returns
        connection = database connection object
    """
    connection = psy.connect(database=db,
                             user=user,
                             password=password,
                             host=host,
                             port=port)
    return connection


def get_col_names(conn, table):
    """ Retrieve the column names for a particular database table.

    Parameters
        conn: database connection object
        table: str = sql database table name

    Returns
        col_names: list = list of table column names
    """

    sql_query = f"""
        SELECT *
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position;
        """

    col_names = []
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        col_data = cursor.fetchone()

        while col_data is not None:
            col_names.append(col_data[3])
            col_data = cursor.fetchone()

    return col_names


def select_table(conn, table):
    """ Retrieves an entire table from the database.

    Parameters
        conn: database connection object
        table: str = table name

    Returns
        : data frame = table retrieved from database
    """
    cols = get_col_names(conn, table)
    sql_query = f"""SELECT * FROM {table}"""
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        data_list = cursor.fetchall()

    return pd.DataFrame(data_list, columns=cols)


def sql_select(conn, table, query):
    """ Execute SQL SELECT command and returns pandas data frame.

    Parameters
        conn: database connection object
        table: str = table name
        query: str = SQL query command

    Returns
        : data frame = table retrieved from database
    """
    cols = get_col_names(conn, table)
    with conn.cursor() as cursor:
        cursor.execute(query)
        data_list = cursor.fetchall()

    return pd.DataFrame(data_list, columns=cols)
