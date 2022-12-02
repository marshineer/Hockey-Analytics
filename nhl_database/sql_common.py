from sqlalchemy import insert


# Reference: Working with data (insert, select, update, delete -> CRUD)
# https://docs.sqlalchemy.org/en/14/tutorial/data.html

def sql_insert(data_iter, table, db_engine):
    """ Inserts a list of dictionaries into an SQL database.

    Reference: https://docs.sqlalchemy.org/en/14/tutorial/data_insert.html

    Parameters
        data_iter: generator object = iterator of csv file row chunks
        table: sqla table object = table in which rows are to be inserted
        db_engine: sqla engine object = database connection engine

    Return
        results: list = metadata related to cursor (insert) operation
    """
    results = []
    with db_engine.connect() as conn:
        for chunk in data_iter:
            result = conn.execute(insert(table), chunk)
            results.append(result)
            conn.commit()

    return results
