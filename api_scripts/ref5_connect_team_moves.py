import os
import pandas as pd
from nhl_api.common import save_nhl_data


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the team data
teams = pd.read_csv(froot + '/../data/teams.csv')
team_list = teams.to_dict('records')

# Define which teams have moved
old_team_ref = {52: 11,  # Atlanta -> Winnipeg
                53: 27}  # Phoenix -> Arizona

# Add the identifier
for team in team_list:
    team_id = team['TeamID']
    team['oldTeamID'] = old_team_ref.get(team_id, None)

# Save the updated team data
save_nhl_data(froot + '/../data/teams.csv', team_list, overwrite=True)
