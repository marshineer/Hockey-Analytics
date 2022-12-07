import os
import pandas as pd
from time import time
from datetime import timedelta
from nhl_api.common import save_nhl_data


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the shift data as a data frame and event data as a list
events_df = pd.read_csv(froot + '/../data/game_events.csv')

# Set the event type order and column priority for sorting
sort_order = ['GIVEAWAY', 'TAKEAWAY', 'HIT', 'GOAL', 'SHOT', 'MISS', 'BLOCK',
              'PENALTY', 'STOP', 'FACEOFF']
sort_priority = ['GameID', 'period', 'periodTime', 'eventTypeId']
events_df.eventTypeId = pd.Categorical(events_df.eventTypeId,
                                       categories=sort_order,
                                       ordered=True)

# Sort the events
t_start = time()
sorted_events = events_df.sort_values(sort_priority)
print(f'It took {timedelta(seconds=(time() - t_start))} to sort '
      f'{len(events_df)} events)')
game_ids = sorted_events.GameID.unique().tolist()

# Renumber the event IDs
t_start = time()
for game_id in game_ids:
    game_mask = sorted_events.loc[sorted_events.GameID == game_id].index.values
    sorted_events.loc[game_mask, 'EventID'] = range(1, len(game_mask) + 1)
print(f'It took {timedelta(seconds=(time() - t_start))} to renumber '
      f'the event IDs')

# Save the updated game event data
event_list = sorted_events.to_dict('records')
save_nhl_data(froot + '/../data/game_events.csv', event_list, overwrite=True)
