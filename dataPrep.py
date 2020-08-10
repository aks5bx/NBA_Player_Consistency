#%%
## Library setup + setting up spark
try:
    spark.stop()
except:
    pass

import findspark
findspark.init()

import pandas as pd 
import pyspark # Call this only after findspark
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import array_contains
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
import sys

sc = SparkContext.getOrCreate() ## optionally takes in a spark conf as a parameter
spark = SparkSession(sc)

#%% 
## Readin play by play data
pbpDF = spark.read.load('playByPlay_2018_19.csv',format="csv", sep=",", inferSchema="true", header="true")
## Register the data as a temporary SQL view if needed
pbpDF.createOrReplaceTempView("pbpVIEW")

## Create a column that is a list of all players on the floor
playerCols = [f.col('HOME_PLAYER_ID_1'), f.col('HOME_PLAYER_ID_2'), f.col('HOME_PLAYER_ID_3'), 
            f.col('HOME_PLAYER_ID_4'), f.col('HOME_PLAYER_ID_5'), f.col('AWAY_PLAYER_ID_1'), f.col('AWAY_PLAYER_ID_2'),
            f.col('AWAY_PLAYER_ID_3'), f.col('AWAY_PLAYER_ID_4'), f.col('AWAY_PLAYER_ID_5')]
pbpDF = pbpDF.withColumn("PlayersOnFloor", f.array(playerCols))

statStuffers = [f.col('ASSIST_PLAYER_ID'), f.col('BLOCK_PLAYER_ID'), f.col('REBOUND_PLAYER_ID'), f.col('SHOT_PLAYER_ID'),
                    f.col('STEAL_PLAYER_ID'), f.col('TURNOVER_PLAYER_ID')]
pbpDF = pbpDF.withColumn("statStuffers", f.array(statStuffers))


## Get all the players (Team 1) 
## TO DO --> Do this for more than just the Player1_Name column
playerList = pbpDF.select('PLAYER1_NAME', 'PLAYER1_ID', 'GAME_ID').distinct().collect()
player = playerList[10].PLAYER1_NAME
playerID = playerList[10].PLAYER1_ID
gameID = playerList[10].GAME_ID

#%%
## For each player and each game
playerGame = pbpDF.filter(pbpDF.GAME_ID == gameID)
playerGame = playerGame.withColumn('IsOnFloor', f.when(array_contains(playerGame['PlayersOnFloor'], playerID), True).otherwise(False))

my_window = Window.partitionBy().orderBy('TIME')
playerGame = playerGame.withColumn('PosDuration', f.lead(playerGame.TIME).over(my_window) - playerGame.TIME)
playerGame = playerGame.withColumn('RestDuration', f.when(playerGame.IsOnFloor == True, 0).otherwise(playerGame.PosDuration))
playerGame = playerGame.withColumn('PlayDuration', f.when(playerGame.IsOnFloor == False, 0).otherwise(playerGame.PosDuration))

playerGame = playerGame.withColumn('TotalRest', f.sum('RestDuration').over(Window.orderBy(f.col('TIME').asc()).rowsBetween(Window.unboundedPreceding, 0)))
playerGame = playerGame.withColumn('TotalPlay', f.sum('PlayDuration').over(Window.orderBy(f.col('TIME').asc()).rowsBetween(Window.unboundedPreceding, 0)))
playerGame = playerGame.withColumn('NetPlay', playerGame.TotalPlay - playerGame.TotalRest)

## Stats Available: Assists, Blocks, ORB, DRB, FGA, Points Scored (FT and FG), FTA, Steal, Turnover, 3PM
playerGame = playerGame.withColumn('AstAdded', f.when(playerGame.ASSIST_PLAYER_ID == playerID, playerGame.ASSIST_COUNT).otherwise(0))
playerGame = playerGame.withColumn('BlockAdded', f.when(playerGame.BLOCK_PLAYER_ID == playerID, playerGame.BLOCK_COUNT).otherwise(0))
playerGame = playerGame.withColumn('ORBAdded', f.when(playerGame.REBOUND_PLAYER_ID == playerID, playerGame.REBOUND_OFFENSIVE_COUNT).otherwise(0))
playerGame = playerGame.withColumn('DRBAdded', f.when(playerGame.REBOUND_PLAYER_ID == playerID, playerGame.REBOUND_DEFENSIVE_COUNT).otherwise(0))
playerGame = playerGame.withColumn('FGAAdded', f.when(playerGame.SHOT_PLAYER_ID == playerID, 1).otherwise(0))
## Extra work required to get points scored
my_window2 = Window.partitionBy().orderBy('TIME').rowsBetween(-sys.maxsize, 0)
playerGame = playerGame.withColumn('FGPoints', f.when((playerGame.SHOT_PLAYER_ID == playerID) & (playerGame.SHOT_MADE == True), playerGame.TOTAL_POINTS_SCORED).otherwise(lit(None)))  
playerGame = playerGame.withColumn('FGPoints', f.last('FGPoints', True).over(Window.partitionBy('GAME_ID').orderBy('TIME').rowsBetween(-sys.maxsize, 0)))
playerGame = playerGame.withColumn('FGPoints', f.when(playerGame.FGPoints.isNull(), 0).otherwise(playerGame.FGPoints))  
playerGame = playerGame.withColumn('FGPtsAdded', playerGame.FGPoints - f.lag(playerGame.FGPoints).over(my_window))
playerGame = playerGame.withColumn('FTPtsAdded', f.when((playerGame.FREE_THROW_PLAYER_ID == playerID) & (playerGame.FREE_THROW_MADE == True), 1).otherwise(0))
playerGame = playerGame.withColumn('TotalPtsAdded', playerGame.FTPtsAdded + playerGame.FGPtsAdded)
## Got Total Points Added, now getting the rest
playerGame = playerGame.withColumn('FTAAdded', f.when((playerGame.FREE_THROW_PLAYER_ID == playerID), 1).otherwise(0))
playerGame = playerGame.withColumn('StealAdded', f.when(playerGame.STEAL_PLAYER_ID == playerID, playerGame.STEAL_COUNT).otherwise(0))
playerGame = playerGame.withColumn('TOAdded', f.when(playerGame.TURNOVER_PLAYER_ID == playerID, playerGame.TURNOVER_COUNT).otherwise(0))
playerGame = playerGame.withColumn('ThreePMAdded', f.when((playerGame.SHOT_PLAYER_ID == playerID) & (playerGame.TotalPtsAdded == 3), 1).otherwise(0))
playerGame = playerGame.withColumn('PFAdded', f.when((playerGame.FOULED_BY_PLAYER_ID == playerID), 1).otherwise(0))

## Making sure that each added column doesn't have nulls
playerGame = playerGame.withColumn('AstAdded', f.when(playerGame.AstAdded.isNull(), 0).otherwise(playerGame.AstAdded))  
playerGame = playerGame.withColumn('BlockAdded', f.when(playerGame.BlockAdded.isNull(), 0).otherwise(playerGame.BlockAdded))  
playerGame = playerGame.withColumn('ORBAdded', f.when(playerGame.ORBAdded.isNull(), 0).otherwise(playerGame.ORBAdded))  
playerGame = playerGame.withColumn('DRBAdded', f.when(playerGame.DRBAdded.isNull(), 0).otherwise(playerGame.DRBAdded))  
playerGame = playerGame.withColumn('FGAAdded', f.when(playerGame.FGAAdded.isNull(), 0).otherwise(playerGame.FGAAdded))  
playerGame = playerGame.withColumn('TotalPtsAdded', f.when(playerGame.TotalPtsAdded.isNull(), 0).otherwise(playerGame.TotalPtsAdded))  
playerGame = playerGame.withColumn('FTAAdded', f.when(playerGame.FTAAdded.isNull(), 0).otherwise(playerGame.FTAAdded))  
playerGame = playerGame.withColumn('StealAdded', f.when(playerGame.StealAdded.isNull(), 0).otherwise(playerGame.StealAdded))  
playerGame = playerGame.withColumn('TOAdded', f.when(playerGame.TOAdded.isNull(), 0).otherwise(playerGame.TOAdded))  
playerGame = playerGame.withColumn('ThreePMAdded', f.when(playerGame.ThreePMAdded.isNull(), 0).otherwise(playerGame.ThreePMAdded))  
playerGame = playerGame.withColumn('PFAdded', f.when(playerGame.PFAdded.isNull(), 0).otherwise(playerGame.PFAdded))  


## Now, calculate a psuedo BPM (https://www.basketball-reference.com/about/bpm2.html#:~:text=Estimate%20a%20regressed%20minutes%20per,%2D%204.75%20%2B%200.175%20*%20ReMPG.)
playerGame = playerGame.withColumn('BPMAdded', f.round((playerGame.TotalPtsAdded * 0.860) + (playerGame.ThreePMAdded * 0.389) + (playerGame.AstAdded * 0.807) +
                        (playerGame.TOAdded * -0.964) + (playerGame.ORBAdded * 0.397) + (playerGame.DRBAdded * 0.1485) + 
                        (playerGame.StealAdded * 1.1885) + (playerGame.BlockAdded * 1.01500) + (playerGame.PFAdded * -0.367) + 
                        (playerGame.FGAAdded * -0.67) + (playerGame.FTAAdded * -0.2945), 5))



playerGameCSV = playerGame.select('GAME_ID', 'TotalRest', 'TotalPlay', 'NetPlay', 'BPMAdded')
playerGameCSV = playerGameCSV.withColumn('PlayerName', lit(player))
playerGameCSV = playerGameCSV.withColumn('PlayerID', lit(playerID))

playerGameCSV.show()

# %%
