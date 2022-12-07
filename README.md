# Hockey Analytics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Goal

## Project Description

## Project Modules

### NHL.com API

### Postgresql Database in Docker
Run these commands from `Hockey-Analytics/nhl_database/`

```shell
    docker compose up -d
    
    python3 db_create_sqla_core.py
```

Where the username, password and database name postgres environment variables are retrieved from a .env file. The username and password are important for connecting to the database through a database tool such as DBeaver. Since the database is hosted locally, and there is nothing sensitive on it, I have used a generic username and password combination.
