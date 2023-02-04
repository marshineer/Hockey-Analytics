import os
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from nhl_api.ref_common import game_time_to_sec
from nhl_api.common import save_nhl_data


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the shift data as a data frame and event data as a list
events_df = pd.read_csv(froot + '/../data/game_events.csv')

# TODO: Is there missing shift data in some games? Or do I just need to drop
#  these? How many games is it? -> Not that many, just remove during analysis
# Set the event type order and column priority for sorting
sort_order = ['GIVEAWAY', 'TAKEAWAY', 'HIT', 'SHOT', 'MISS', 'BLOCK', 'GOAL',
              'PENALTY', 'STOP', 'FACEOFF']
sort_priority = ['GameID', 'period', 'periodTime', 'eventTypeId']
events_df.eventTypeId = pd.Categorical(events_df.eventTypeId,
                                       categories=sort_order,
                                       ordered=True)

# Sort the events
t_start = time()
sorted_events = events_df.sort_values(sort_priority)
print(f'It took {timedelta(seconds=(time() - t_start))} to sort '
      f'{len(events_df)} events')
game_ids = sorted_events.GameID.unique().tolist()

# Renumber the event IDs
t_start = time()
for game_id in game_ids:
    game_mask = sorted_events.loc[sorted_events.GameID == game_id].index.values
    sorted_events.loc[game_mask, 'EventID'] = range(1, len(game_mask) + 1)
print(f'It took {timedelta(seconds=(time() - t_start))} to renumber '
      f'the event IDs')

# Check for stoppage events that are not followed by a faceoff
t_start = time()
event_list = sorted_events.to_dict('records')
new_event_list = []
rm_inds = []
last_event = None
last_type = None
last_time = 0
dbl_stop = False
skip_next = False
for i, event_x in enumerate(event_list):
    event_type = event_x['eventTypeId']
    event_time = game_time_to_sec(event_x['periodTime'])
    # if event_type == 'FACEOFF' and ~(last_type == 'STOP' or period_time == 0):
    #     next_event = event_list[i + 1].copy()
    #     next_time = game_time_to_sec(next_event['periodTime'])
    #     next_type = next_event['eventTypeId']
    #     if next_type == 'STOP' and (next_time - period_time) == 1:
    #         switch_next = True
    # elif event_type in ['SHOT', 'MISS', 'BLOCK', 'GOAL']:
    if event_type == 'STOP' and not skip_next:
        next_event = event_list[i + 1].copy()
        next_time = game_time_to_sec(next_event['periodTime'])
        next_type = next_event['eventTypeId']
        if next_type == 'STOP' and next_time == event_time:
            dbl_stop = True
            last_event = event_x.copy()
            last_type = event_type
            last_time = event_time
            new_event_list.append(event_x.copy())
            continue
        elif next_type != 'FACEOFF':
            next2_event = event_list[i + 2].copy()
            next2_time = game_time_to_sec(next2_event['periodTime'])
            next2_type = next2_event['eventTypeId']
            # if dbl_stop:
            #     last2_event = event_list[i - 2]
            #     last2_type = last2_event['eventTypeId']
            #     last2_time = game_time_to_sec(last2_event['periodTime'])
            #     if last2_type == 'FACEOFF' and (event_time - last2_time) == 1:
            if last_type == 'FACEOFF' and (event_time - last_time) < 5:
                # if event_list[i - 2]['eventTypeId'] == 'STOP':
                #     continue
                temp_event = event_x.copy()
                event_list[i] = last_event.copy()
                event_list[i]['EventID'] = temp_event['EventID']
                event_list[i]['periodTime'] = temp_event['periodTime']
                event_list[i - 1] = temp_event.copy()
                event_list[i - 1]['EventID'] = last_event['EventID']
                event_list[i - 1]['periodTime'] = last_event['periodTime']
            #     new_event_list[-1] = temp_event.copy()
            #     new_event_list[-1]['EventID'] = last_event['EventID']
            #     new_event_list[-1]['periodTime'] = last_event['periodTime']
            #     new_event_list.append(last_event.copy())
            #     new_event_list[-1]['EventID'] = temp_event['EventID']
            #     new_event_list[-1]['periodTime'] = temp_event['periodTime']
            elif next2_type == 'FACEOFF' and 0 < (next2_time - event_time) < 3:
                temp_event = event_x.copy()
                event_list[i] = next_event.copy()
                event_list[i]['EventID'] = temp_event['EventID']
                event_list[i]['periodTime'] = temp_event['periodTime']
                event_list[i + 1] = temp_event.copy()
                event_list[i + 1]['EventID'] = next_event['EventID']
                event_list[i + 1]['periodTime'] = next_event['periodTime']
                skip_next = True
            else:
                rm_inds.append(i)
    # elif (np.isnan(event_x['xCoord']) or np.isnan(event_x['yCoord'])) and \
    #         event_type != 'PENALTY':
    #     rm_inds.append(i)
    skip_next = False
    dbl_stop = False
    last_event = event_x.copy()
    last_type = event_type
    last_time = event_time
    # new_event_list.append(event_x.copy())
print(f'It took {timedelta(seconds=(time() - t_start))} to fix stop events')
print(f'{len(rm_inds)} stop events were removed')
# print(pd.DataFrame(event_list).eventTypeId.iloc[rm_inds].value_counts())
# print(rm_inds)
# print(pd.DataFrame(event_list).eventTypeId.iloc[rm_inds].unique().tolist())
all_events = [event_list[i] for i in range(len(event_list)) if i not in rm_inds]

# Save the updated game event data
save_nhl_data(froot + '/../data/game_events.csv', event_list, overwrite=True)
