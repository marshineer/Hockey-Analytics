import os
from sqlalchemy import create_engine
from nhl_database.db_models_core import teams, players, coaches, games, shifts, \
    game_events, shots, team_box_scores, skater_box_scores, goalie_box_scores, \
    skater_seasons, goalie_seasons
from nhl_database.db_models_core import meta
from nhl_database.db_common import get_file_names, generator_from_csv
from nhl_database.sql_common import sql_insert

# Initiate a database connection engine
DATABASE_URL = 'postgresql+psycopg2://marshineer:password@localhost:5432/nhl_db'
db_engine = create_engine(DATABASE_URL, echo=True)

# Connect to the database
connection = db_engine.connect()

# Create the tables ("Emit DDL to the DB")
meta.create_all(db_engine)

# Load the data
froot = str(os.path.dirname(__file__))
fpath = froot + '/../data/'
data_files = get_file_names(fpath)
print(data_files)
tables = [games, shots, skater_seasons, team_box_scores, coaches,
          goalie_box_scores, shifts, skater_box_scores, goalie_seasons,
          teams, players, game_events]
for file, table in zip(data_files, tables):
    # Read the data and convert it into a chunked iterator
    data_iter = generator_from_csv(fpath + file, read_chunk=100, iter_chunk=100)

    # Insert the chunks into the database
    results = sql_insert(data_iter, table, db_engine)
