# Hockey Analytics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Outline
```mermaid
    graph BT;
    Expected Goals --> Player Impact;
    In-game Win Probability --> Player Impact;
    Player Impact --> Team Success;
```

Add a flow chart of models and their relationships
- Team success
    - Player effectiveness/impact
        - In-game win probability
        - Expected goals

Add descriptions of each model (including inputs/outputs, model use, and thought process/notes)

In-Game Win Probability: Evaluates the probability a team will win a particular game based on the current game state (score, time, etc...). Used to determine whether players commonly impact the game during important moments, or run up points when the outcome of the game is no longer in question

Expected goals: Evaluate the offensive contributions of a player by how many scoring opportunities they create (opportunities are more conducive to statistical analysis since scoring events are relatively rare). Used to determine the impact of a player. Whether players consistently outperform their expected goal totals is an indication of the individual player's talent (and an indication that the xG model does not capture all important info).

Player impact: Evaluate the overall impact of a player on an individual game. Takes into account the player's average scoring/defensive contributions, as well as their tendency to contribute at important times.

Team success: Evaluate the probability that a team will be successful in individual games and across seasons. Takes into account the aggregation of player contributions on the team, and how the mix of player attributes affects the overall team effectiveness. Also accounts for a team's recent success, travel, etc...

## Project Description

## Project Modules
List of modules:
- API
- Database
- Modelling

### NHL.com API

### Postgresql Database in Docker
Run these commands from `Hockey-Analytics/nhl_database/`

```shell
    docker compose up -d
    
    python3 db_create_sqla_core.py
```

Where the username, password and database name postgres environment variables are retrieved from a .env file. The username and password are important for connecting to the database through a database tool such as DBeaver. Since the database is hosted locally, and there is nothing sensitive on it, I have used a generic username and password combination.

