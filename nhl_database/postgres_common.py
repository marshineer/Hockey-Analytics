from os import listdir
from os.path import isfile, join


def store_data(data, table_name, engine):
    """ Stores the data from a pandas data frame in a database

    Parameters
        data: DataFrame =
        table_name: str =
        engine: engine object =
    """
    data.to_sql(name=table_name, con=engine, if_exists='append', index=False)


def get_file_names(dpath):
    """ Returns a list of all files in a directory.

    The function is non-recursive and ignores subdirectories.
    https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

    Parameters
        dpath: str = path to the directory of interest

    Returns
        : list = all files in the directory
    """
    return [f for f in listdir(dpath) if isfile(join(dpath, f))]


