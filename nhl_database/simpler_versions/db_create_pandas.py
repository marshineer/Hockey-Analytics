import os
import pandas as pd
from sqlalchemy import create_engine
from nhl_database.db_common import get_file_names


# Note: this solution does not create table relationships or primary/foreign keys
# Initiate a database connection engine
DATABASE_URL = 'postgresql+psycopg2://marshineer:password@localhost:5432/nhl_db'
db_engine = create_engine(DATABASE_URL, echo=True)

# Dynamically set the CWD
froot = str(os.path.dirname(__file__))
fpath = froot + '/../../data/'
tables = ['games', 'shots', 'skater_seasons', 'team_boxscores', 'coaches',
          'goalie_boxscores', 'shifts', 'skater_boxscores', 'goalie_seasons',
          'teams', 'players', 'game_events']
data_files = get_file_names(fpath)
print(data_files)
for file, name in zip(data_files, tables):
    data = pd.read_csv(fpath + file)
    data.to_sql(name=name, con=db_engine, if_exists='append', index=False)
