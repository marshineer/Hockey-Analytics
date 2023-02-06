import pandas as pd
import psycopg2 as psy


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


def sql_select(conn, query, table=None, cols=None):
    """ Execute SQL SELECT command and returns pandas dataframe.

    Parameters
        conn: database connection object
        table: str = table name
        query: str = SQL query command

    Returns
        : dataframe = table retrieved from database
    """

    if table is not None:
        cols = get_col_names(conn, table)
    with conn.cursor() as cursor:
        cursor.execute(query)
        data_list = cursor.fetchall()

    if cols is not None:
        return pd.DataFrame(data_list, columns=cols)
    else:
        return pd.DataFrame(data_list)


def sql_join(table1, table2, join_on, join_type, cols1=None, cols2=None,
             conn=None, return_command=False):
    """ Executes or returns the string for an SQL JOIN query.

    If executing, the result is returned as a pandas dataframe. If return_command
    is set to True, the SQL command string is returned instead.

    Parameters
        table1: str = left table name
        table2: str = right table name
        join_on: str = the key on which to join the tables
        cols1: list = columns to be selected from table 1
        cols2: list = columns to be selected from table 2
        conn: database connection object
        return_command: bool = whether to return the joined table or query string

    Returns
        joined_df: dataframe = result of JOIN query
        sql_query: str = SQL command string
    """

    # Set the columns to select
    if cols1 is None and cols2 is None:
        select_str = '*'
    elif cols2 is None:
        select_str = ''
        for col in cols1:
            select_str += f'{table1}.{col}, '
        select_str = select_str[:-2]
    elif cols1 is None:
        select_str = ''
        for col in cols2:
            select_str += f'{table2}.{col}, '
        select_str = select_str[:-2]
    else:
        select_str = ''
        for col in cols1:
            select_str += f'{table1}.{col}, '
        for col in cols2:
            select_str += f'{table2}.{col}, '
        select_str = select_str[:-2]

    # Set which key to join the tables on
    join_on_str = f"{table1}.{join_on} = {table2}.{join_on}"

    # Create the SQL query
    sql_query = f"""SELECT {select_str} FROM {table1} 
                    {join_type} JOIN {table2} ON {join_on_str}"""

    if return_command:
        return sql_query
    else:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            data_list = cursor.fetchall()
        cols = cols1.append(cols2)
        return pd.DataFrame(data_list, columns=cols)
