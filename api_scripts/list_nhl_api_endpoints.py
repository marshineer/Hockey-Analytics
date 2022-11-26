from nhl_api.api_common import request_json


# Define the NHL.com API url
API_URL = 'https://statsapi.web.nhl.com/api/v1'

# Request and print a list of alternate endpoints for NHL.com
endpoints = request_json(API_URL + '/configurations')
print(f'A list of alternate endpoints:')
for endpoint in endpoints:
    print(f"{endpoint['description']}: {endpoint['endpoint']}")
print()

# Print a list of all the possible event types
event_list = request_json(API_URL + '/playTypes')
print(f'A full list of game event types:')
for event in event_list:
    print(f"{event['name']}")
print()

# Print a list of all the possible player types during events
player_list = request_json(API_URL + '/playTypesPlayer')
print(f'A full list of player types during game events:')
for playerType in player_list:
    print(playerType)
print()

# Print a list of all the possible player types during events
player_list = request_json(API_URL + '/gameTypes')
print(f'A full list of game types during game events:')
for playerType in player_list:
    print(playerType)
print()
