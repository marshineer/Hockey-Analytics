import pandas as pd
from os import listdir
from os.path import isfile, join
from itertools import zip_longest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nhl_database.db_models_orm import BaseClass
from nhl_database.db_config import DATABASE_URL


# Initiate a database connection engine
db_engine = create_engine(DATABASE_URL, echo=True)

# Initiate a session for interacting with the database
Session = sessionmaker(bind=db_engine)
# # Instantiate an individual session (do this whenever we need to
# s = Session()
# s.close()


def get_file_names(dpath):
    """ Returns a list of all files in a directory.

    The function is non-recursive and ignores subdirectories.
    https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

    Parameters
        dpath: str = path to the directory of interest

    Returns
        : list = all files in the directory
    """
    return [file for file in listdir(dpath) if isfile(join(dpath, file))]


def grouper(iterable, chunk_sz, fillvalue=None):
    """ Separates an iterable into non-overlapping, fixed-length chunks.

    Modified from the recipes section of the itertools documentation.
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    If the chunks do not divide evenly into the iterable length, the remaining
    spots are filled.

    Parameters
        iterable: iterator = object to be chunked
        chunk_sz: int = size of each chunk
        fillvalue: any = value to fill extra spots in chunk

    Returns
        : iterator = chunked generat
    """
    args = [iter(iterable)] * chunk_sz
    return zip_longest(*args, fillvalue=fillvalue)


def generator_from_csv(fpath, read_chunk=100, iter_chunk=100):
    """ Returns an iterator that will cycle through all rows in a csv file

    Reference: https://realpython.com/introduction-to-python-generators/
    Also: https://stackoverflow.com/questions/17444679/reading-a-huge-csv-file

    Parameters
        fpath: str = relative path to file
        read_chunk: int = chunk size for reading data
        iter_chunk: int = chunk size for creating iterator

    Returns
        data_iter: generator object = iterator of csv file row chunks
    """
    data_df = pd.read_csv(fpath, chunksize=read_chunk)
    data_list = data_df.to_dict('records')  # List of dictionaries
    data_iter = grouper(data_list, iter_chunk, fillvalue=None)
    return data_iter


# Reference
# https://www.learndatasci.com/tutorials/using-databases-python-postgres-sqlalchemy-and-alembic/
# https://dev.to/spaceofmiah/ddl-on-sqlalchemy-core-api-5gpe
def recreate_database():
    BaseClass.metadata.drop_all(db_engine)
    BaseClass.metadata.create_all(db_engine)


def create_database():
    BaseClass.metadata.create_all(db_engine)
