from sqlalchemy import MetaData, Table, Column
from sqlalchemy import Integer, String, Float, Boolean, DateTime, ForeignKey


meta = MetaData()

teams = Table(
    'teams', meta,
    Column('team_id', Integer, primary_key=True),
    Column('abbreviation', String),
    Column('team_location', String),
    Column('team_name', String, nullable=False),
    Column('arena_city', String),
    Column('arena_name', String),
    Column('arena_latitude', Float),
    Column('arena_longitude', Float),
    Column('old_team_id', Integer),
)

players = Table(
    'players', meta,
    Column('player_id', Integer, primary_key=True),
    Column('full_name', String, nullable=False),
    Column('birth_date', String),
    Column('nationality', String),
    Column('position', String),
    Column('position2', String),
    Column('rookie_season', Integer),
    Column('shoots_catches', String),
    Column('height_cm', Float),
    Column('weight_kg', Float),
)

coaches = Table(
    'coaches', meta,
    Column('coach_id', Integer, primary_key=True),
    Column('full_name', String, nullable=False),
    Column('title', String),
    Column('abbreviation', String),
)

games = Table(
    'games', meta,
    Column('game_id', Integer, primary_key=True),
    Column('season', String),
    Column('type', String),
    Column('home_team_id', ForeignKey('teams.team_id'), nullable=False),
    Column('away_team_id', ForeignKey('teams.team_id'), nullable=False),
    Column('home_coach_id', ForeignKey('coaches.coach_id')),
    Column('away_coach_id', ForeignKey('coaches.coach_id')),
    Column('home_score', Integer),
    Column('away_score', Integer),
    Column('home_win', Boolean),
    Column('number_periods', Integer),
    Column('overtime', Boolean),
    Column('shootout', Boolean),
    Column('location', String),
    Column('arena', String),
    Column('datetime', DateTime),
    Column('time_zone', String),
    Column('time_zone_offset', Integer),
    Column('active_home_players', String),
    Column('active_away_players', String),
)

shifts = Table(
    'shifts', meta,
    Column('game_id', ForeignKey('games.game_id'), primary_key=True),
    Column('player_id', ForeignKey('players.player_id'), primary_key=True),
    Column('shift_id', Integer, primary_key=True),
    Column('period', Integer),
    Column('start_time', String),
    Column('end_time', String),
    Column('duration', String),
)

game_events = Table(
    'game_events', meta,
    Column('game_id', ForeignKey('games.game_id'), primary_key=True),
    Column('event_id', Integer, primary_key=True),
    Column('event_type_id', String, nullable=False),
    Column('secondary_type', String),
    Column('description', String),
    Column('home_team_id', ForeignKey('teams.team_id')),
    Column('away_team_id', ForeignKey('teams.team_id')),
    Column('player1_id', ForeignKey('players.player_id')),
    Column('player1_type', String),
    Column('player1_home', Boolean),
    Column('player2_id', ForeignKey('players.player_id')),
    Column('player2_type', String),
    Column('period', Integer),
    Column('period_type', String),
    Column('period_time', String),
    Column('home_score', Integer),
    Column('away_score', Integer),
    Column('x_coord', Integer),
    Column('y_coord', Integer),
    Column('assist1_id', ForeignKey('players.player_id')),
    Column('assist2_id', ForeignKey('players.player_id')),
    Column('empty_net', Boolean),
    Column('pim', Integer),
    Column('players_home', String),
    Column('players_away', String),
)

shots = Table(
    'shots', meta,
    Column('game_id', ForeignKey('games.game_id'), primary_key=True),
    Column('shot_id', Integer, primary_key=True),
    Column('shooter_id', ForeignKey('players.player_id'), nullable=False),
    Column('assist1_id', Integer),
    Column('assist2_id', Integer),
    Column('shot_type', String),
    Column('shot_result', String),
    Column('period', Integer),
    Column('period_time', String),
    Column('shot_time', Integer),
    Column('home_team_id', ForeignKey('teams.team_id')),
    Column('away_team_id', ForeignKey('teams.team_id')),
    Column('home_score', Integer),
    Column('away_score', Integer),
    Column('goal_lead_prior', Integer),
    Column('home_shots', Integer),
    Column('away_shots', Integer),
    Column('shot_lead_prior', Integer),
    Column('x_coord', Integer),
    Column('y_coord', Integer),
    Column('net_distance', Float),
    Column('net_angle', Float),
    Column('last_event_type', String),
    Column('time_since_last', String),
    Column('last_same_end', Boolean),
    Column('last_same_team', Boolean),
    Column('last_x_coord', Integer),
    Column('last_y_coord', Integer),
    Column('delta_x', Float),
    Column('delta_y', Float),
    Column('dist_change', Float),
    Column('angle_change', Float),
    Column('last_turnover', Boolean),
    Column('rebound_shot', Boolean),
    Column('play_ends', Boolean),
    Column('puck_frozen', Boolean),
    Column('goal', Boolean),
    Column('missed', Boolean),
    Column('blocked', Boolean),
    Column('empty_net', Boolean),
    Column('shooter_home', Boolean),
    Column('shooter_age', Float),
    Column('shooter_seasons', Integer),
    Column('shooter_hand', String),
    Column('shooter_position', String),
    Column('off_wing_shot', Boolean),
    Column('players_home', String),
    Column('players_away', String),
)

team_boxscores = Table(
    'team_boxscores', meta,
    Column('game_id', ForeignKey('games.game_id'), primary_key=True),
    Column('team_id', ForeignKey('teams.team_id'), primary_key=True),
    Column('home_team', Boolean, nullable=False),
    Column('goals', Integer),
    Column('shots', Integer),
    Column('hits', Integer),
    Column('blocks', Integer),
    Column('pim', Integer),
    Column('ppg', Integer),
    Column('power_plays', Integer),
    Column('faceoff_pct', Float),
    Column('takeaways', Integer),
    Column('giveaways', Integer),
)

skater_boxscores = Table(
    'skater_boxscores', meta,
    Column('game_id', ForeignKey('games.game_id'), primary_key=True),
    Column('player_id', ForeignKey('players.player_id'), primary_key=True),
    Column('home_team', Boolean, nullable=False),
    Column('goals', Integer),
    Column('assists', Integer),
    Column('shots', Integer),
    Column('hits', Integer),
    Column('blocks', Integer),
    Column('pim', Integer),
    Column('plus_minus', Integer),
    Column('faceoff_wins', Integer),
    Column('faceoff_taken', Integer),
    Column('takeaways', Integer),
    Column('giveaways', Integer),
    Column('ppg', Integer),
    Column('ppa', Integer),
    Column('shg', Integer),
    Column('sha', Integer),
    Column('even_toi', String),
    Column('pp_toi', String),
    Column('sh_toi', String),
    Column('total_toi', String),
)

goalie_boxscores = Table(
    'goalie_boxscores', meta,
    Column('game_id', ForeignKey('games.game_id'), primary_key=True),
    Column('player_id', ForeignKey('players.player_id'), primary_key=True),
    Column('home_team', Boolean, nullable=False),
    Column('decision', String),
    Column('saves', Integer),
    Column('shots', Integer),
    Column('saves_even', Integer),
    Column('shots_even', Integer),
    Column('saves_pp', Integer),
    Column('shots_pp', Integer),
    Column('saves_sh', Integer),
    Column('shots_sh', Integer),
    Column('goals', Integer),
    Column('assists', Integer),
    Column('pim', Integer),
    Column('total_toi', String),
)

skater_seasons = Table(
    'skater_seasons', meta,
    Column('player_id', ForeignKey('players.player_id'), primary_key=True),
    Column('full_name', String, nullable=False),
    Column('team_id', Integer),
    Column('team', String, nullable=False, primary_key=True),
    Column('league', String, nullable=False, primary_key=True),
    Column('season', Integer, nullable=False, primary_key=True),
    Column('games', Integer),
    Column('goals', Integer),
    Column('assists', Integer),
    Column('points', Integer),
    Column('shots', Integer),
    Column('hits', Integer),
    Column('blocks', Integer),
    Column('pim', Integer),
    Column('plus_minus', Integer),
    Column('faceoff_pct', Float),
    Column('ppg', Integer),
    Column('ppp', Integer),
    Column('shg', Integer),
    Column('shp', Integer),
    Column('gwg', Integer),
    Column('otg', Integer),
    Column('even_toi', String),
    Column('pp_toi', String),
    Column('sh_toi', String),
)

goalie_seasons = Table(
    'goalie_seasons', meta,
    Column('player_id', ForeignKey('players.player_id'), primary_key=True),
    Column('full_name', String, nullable=False),
    Column('team_id', Integer),
    Column('team', String, nullable=False, primary_key=True),
    Column('league', String, nullable=False, primary_key=True),
    Column('season', Integer, nullable=False, primary_key=True),
    Column('games', Integer),
    Column('starts', Integer),
    Column('wins', Integer),
    Column('losses', Integer),
    Column('saves', Integer),
    Column('shots', Integer),
    Column('shutouts', Integer),
    Column('gaa', Integer),
    Column('saves_even', Integer),
    Column('shots_even', Float),
    Column('saves_pp', Integer),
    Column('shots_pp', Integer),
    Column('saves_sh', Integer),
    Column('shots_sh', Integer),
    Column('toi', String),
)
