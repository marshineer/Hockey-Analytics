event_cols = {'secondaryType': None,
              'player1ID': None,
              'player1Type': None,
              'player1Home': None,
              'player2ID': None,
              'player2Type': None,
              'assist1ID': None,
              'assist2ID': None,
              'emptyNet': None,
              'xCoord': None,
              'yCoord': None,
              'PIM': None}

skater_stat_cols = {'games': None,
                    'goals': None,
                    'assists': None,
                    'points': None,
                    'shots': None,
                    'hits': None,
                    'pim': None,
                    'blocked': None,
                    'plusMinus': None,
                    'faceOffPct': None,
                    'gameWinningGoals': None,
                    'overTimeGoals': None,
                    'powerPlayGoals': None,
                    'powerPlayPoints': None,
                    'shortHandedGoals': None,
                    'shortHandedPoints': None,
                    'evenTimeOnIce': None,
                    'powerPlayTimeOnIce': None,
                    'shortHandedTimeOnIce': None}

goalie_stat_cols = {'games': None,
                    'gamesStarted': None,
                    'wins': None,
                    'losses': None,
                    'shutouts': None,
                    'shotsAgainst': None,
                    'saves': None,
                    'evenShots': None,
                    'powerPlayShots': None,
                    'shortHandedShots': None,
                    'evenSaves': None,
                    'powerPlaySaves': None,
                    'shortHandedSaves': None,
                    'timeOnIce': None}

shot_types = {'Wrist Shot': 'WRIST',
              'Snap Shot': 'SNAP',
              'Slap Shot': 'SLAP',
              'Backhand': 'BACKHAND',
              'Tip-In': 'TIP',  # Intentional, by scoring team
              'Deflected': 'DEFLECT',  # Unintentional, off of someone
              'Wrap-around': 'WRAP'}

penalty_names = {'Hooking': 'HOOK',
                 'Tripping': 'TRIP',
                 'Elbowing': 'ELBOW',
                 'Slashing': 'SLASH',
                 'Holding': 'HOLD',
                 'Holding the stick': 'HOLD_STICK',
                 'Interference': 'INTERFERENCE',
                 'Interference - Goalkeeper': 'INTERFERENCE_GOALIE',
                 'Cross checking': 'CROSS_CHECK',
                 'Cross check - double minor': 'CROSS_CHECK',
                 'Hi-sticking': 'HIGH_STICK',
                 'Hi stick - double minor': 'HIGH_STICK',
                 'Roughing': 'ROUGH',
                 'Fighting': 'FIGHT',
                 'Instigator': 'INSTIGATOR',
                 'Instigator - face shield': 'INSTIGATOR',
                 'Instigator - Misconduct': 'INSTIGATOR',
                 'Charging': 'CHARGE',
                 'Boarding': 'BOARD',
                 'Checking from behind': 'HIT_FROM_BEHIND',
                 'Illegal check to head': 'HEAD_CHECK',
                 'Diving': 'DIVE',
                 'Embellishment': 'DIVE',
                 'Kneeing': 'KNEE',
                 'Clipping': 'CLIP',
                 'Spearing': 'SPEAR',
                 'Spearing - double minor': 'SPEAR',
                 'Butt ending': 'BUTT_END',
                 'Butt ending - double minor': 'BUTT_END',
                 'Head butting': 'HEAD_BUTT',
                 'Head butting - double minor': 'HEAD_BUTT',
                 'Misconduct': 'MISCONDUCT_10_MIN',
                 'Game misconduct': 'MISCONDUCT_GAME',
                 'Game Misconduct - Team staff': 'MISCONDUCT_GAME',
                 'Game Misconduct - Head coach': 'MISCONDUCT_GAME',
                 'Match penalty': 'MATCH',
                 'Unsportsmanlike conduct': 'UNSPORTSMANLIKE',
                 'Too many men on the ice': 'TOO_MANY_MEN',
                 'Bench': 'BENCH',
                 'Illegal substitution': 'BENCH',
                 'Coach or Manager on the ice': 'BENCH',
                 'Delaying Game - Puck over glass': 'PUCK_OVER_GLASS',
                 'Delaying Game - Illegal play by goalie': 'DELAY_OF_GAME',
                 'Delaying Game - Smothering puck': 'DELAY_OF_GAME',
                 'Delay Gm - Face-off Violation': 'DELAY_OF_GAME',
                 'Delaying the game': 'DELAY_OF_GAME',
                 'Delay of game': 'DELAY_OF_GAME',
                 'Closing hand on puck': 'DELAY_OF_GAME',
                 'Face-off violation': 'DELAY_OF_GAME',
                 'Throwing stick': 'OTHER',
                 'Illegal stick': 'OTHER',
                 'Broken stick': 'OTHER',
                 'Playing without a helmet': 'OTHER',
                 'Objects on ice': 'OTHER',
                 'Goalie leave crease': 'OTHER',
                 "Leaving player's/penalty bench": 'OTHER',
                 'Abuse of Officials': 'OTHER',
                 'Abusive language': 'OTHER',
                 'Aggressor': 'OTHER',
                 'Illegal equipment': 'OTHER',
                 'Minor': 'OTHER',
                 'Major': 'OTHER',
                 'Player leaves bench': 'OTHER',
                 'Leaving penalty box': 'OTHER',
                 'Concealing puck': 'OTHER',
                 'Missing key [PD_151]': 'OTHER',
                 'Missing key [PD_152]': 'OTHER',
                 'Missing key [PD_161]': 'OTHER',
                 'Missing key [PD_162]': 'OTHER',
                 'Missing key [PD_149]': 'OTHER',
                 'Missing key [PD_150]': 'OTHER',
                 # 'PS - Hooking on breakaway': 'HOOK',
                 # 'PS - Goalkeeper displaced net': 'DELAY_OF_GAME',
                 # 'PS - Slash on breakaway': 'SLASH',
                 # 'PS - Covering puck in crease': 'DELAY_OF_GAME',
                 # 'PS - Holding stick on breakaway': 'HOLD_STICK',
                 # 'PS - Thow object at puck': 'OTHER',
                 # 'PS - Holding on breakaway': 'HOLD',
                 # 'PS - Net displaced': 'DELAY_OF_GAME',
                 # 'PS - Tripping on breakaway': 'TRIP',
                 'PS - Hooking on breakaway': 'PENALTY_SHOT',
                 'PS - Goalkeeper displaced net': 'PENALTY_SHOT',
                 'PS - Slash on breakaway': 'PENALTY_SHOT',
                 'PS - Covering puck in crease': 'PENALTY_SHOT',
                 'PS - Holding stick on breakaway': 'PENALTY_SHOT',
                 'PS - Thow object at puck': 'PENALTY_SHOT',
                 'PS - Holding on breakaway': 'PENALTY_SHOT',
                 'PS - Net displaced': 'PENALTY_SHOT',
                 'PS - Tripping on breakaway': 'PENALTY_SHOT'}
