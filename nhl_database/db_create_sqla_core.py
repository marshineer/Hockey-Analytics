import os
import json
from sqlalchemy import create_engine, insert
from time import time
from datetime import timedelta
from nhl_database.db_models_core import teams, players, coaches, games, shifts, \
    game_events, shots, team_boxscores, skater_boxscores, goalie_boxscores, \
    skater_seasons, goalie_seasons
from nhl_database.db_models_core import meta
from nhl_database.db_common import generator_from_csv


# Initiate a database connection engine
froot = str(os.path.dirname(__file__))
with open(froot + '/../db_config.json') as f:
    config = json.load(f)
DATABASE_URL = config['db_url']
db_engine = create_engine(DATABASE_URL, echo=False)

# Connect to the database
connection = db_engine.connect()

# Create the tables ("Emit DDL to the DB")
t0 = time()
meta.drop_all(db_engine)
print(f"It took {timedelta(seconds=(time() - t0))} to delete the old tables ")
t0 = time()
meta.create_all(db_engine)
print(f"It took {timedelta(seconds=(time() - t0))} to create the new tables ")

# Load the data
froot = str(os.path.dirname(__file__))
fpath = froot + '/../data/'
# fpath = froot + '/../data/2010-21_backup/backup_formatted/'

# Set the order the table are to be populated in
tables = [teams, players, coaches, games, shifts, game_events, shots,
          team_boxscores, skater_boxscores, goalie_boxscores,
          skater_seasons, goalie_seasons]

for table in tables:
    # The tables and data files must have the same names
    t_start = time()
    file = str(table.description) + '.csv'

    # Read the data and convert it into a chunked iterator
    data_iter = generator_from_csv(fpath + file, iter_chunk=1000)

    # Insert the chunks into the database
    with db_engine.connect() as conn:
        for chunk in data_iter:
            result = conn.execute(insert(table), chunk)

    print(f"It took {timedelta(seconds=(time() - t_start))} to insert all data "
          f"into the '{table.description}' table")
