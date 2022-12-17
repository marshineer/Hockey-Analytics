import os
import pandas as pd
from time import time
from datetime import timedelta
from nhl_api.common import save_nhl_data


froot = str(os.path.dirname(__file__))
shifts_df = pd.read_csv(froot + f'/../data/shifts.csv')
shifts_df.dropna(inplace=True)
shifts_df.reset_index(drop=True, inplace=True)
key_list = ['GameID', 'PlayerID', 'ShiftID']
duplicate_shifts = shifts_df[shifts_df.duplicated(key_list, keep='last')]
duplicate_inds = duplicate_shifts.index.tolist()
# print(len(duplicate_inds))

t_start = time()
for ind in duplicate_inds:
    shift = shifts_df.iloc[ind]
    game_id = shift.GameID
    player_id = shift.PlayerID
    shift_id = shift.ShiftID
    shifts_df.loc[(shifts_df.GameID == game_id) &
                  (shifts_df.PlayerID == player_id) &
                  (shifts_df.ShiftID >= shift_id), 'ShiftID'] += 1
    shifts_df.loc[ind, 'ShiftID'] = shift_id

duplicate_shifts2 = shifts_df[shifts_df.duplicated(key_list)]
# print(len(duplicate_shifts2))
print(f'\nTook {timedelta(seconds=(time() - t_start))} to renumber shifts')
shift_list = shifts_df.to_dict('records')
save_nhl_data(froot + f'/../data/shifts.csv', shift_list, overwrite=True)
