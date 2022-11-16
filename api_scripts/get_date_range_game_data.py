import csv
from os.path import exists
from time import time, sleep
import datetime
from nhl_api.api_parsers import get_range_data, request_json


# Specify the date range
start_year = 2010
start_date = f'{start_year}-09-01'
if start_year == 2016:
    start_date = '2016-09-30'
elif start_year == 2020:
    start_date = '2021-01-01'
elif start_year == 2021:
    start_date = '2021-11-01'

end_year = 2010
end_date = f'{end_year + 1}-08-31'
if end_year == 2020:
    end_date[-5:] = '2020-10-31'
end_date = '2010-11-01'

# Check the end date is after the start date
# https://stackoverflow.com/questions/20365854/comparing-two-date-strings-in-python

# Request game data for the specified range
API_URL_RANGE = f'https://statsapi.web.nhl.com/api/v1/' \
                f'schedule?startDate={start_date}&endDate={end_date}'
game_dict = request_json(API_URL_RANGE)
