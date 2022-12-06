import pandas as pd
from more_itertools import chunked
from numpy import nan


def generator_from_csv(fpath, iter_chunk=100):
    """ Returns an iterator that will cycle through all rows in a csv file

    Reference: https://realpython.com/introduction-to-python-generators/
    Also: https://stackoverflow.com/questions/17444679/reading-a-huge-csv-file

    Parameters
        fpath: str = relative path to file
        iter_chunk: int = chunk size for creating iterator

    Returns
        data_iter: generator object = iterator of csv file row chunks
    """
    data_df = pd.read_csv(fpath)
    data_df = data_df.replace(nan, None)
    data_list = data_df.to_dict('records')  # List of dictionaries
    data_iter = chunked(data_list, iter_chunk)
    return data_iter
