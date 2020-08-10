
#%%
import pandas as pd 
pbp = pd.read_csv('playByPlay_2018_19.csv')

subset = pbp[pbp.GAME_ID == 21801052]
subset = pbp[['FREE_THROW_PLAYER_ID', 'FREE_THROW_MADE', 'SHOT_PLAYER_ID', 'TOTAL_POINTS_SCORED', 'SHOT_MADE']]