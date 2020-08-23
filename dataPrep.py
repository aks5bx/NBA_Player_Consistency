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
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sc = SparkContext.getOrCreate() ## optionally takes in a spark conf as a parameter
spark = SparkSession(sc)








#%%





pbpDF = pd.read_csv('playByPlay_2018_19.csv')
playerList = pbpDF[['PLAYER1_NAME', 'PLAYER1_ID', 'GAME_ID']].drop_duplicates()

#%% 
## Readin play by play data
pbpDF = spark.read.load('playByPlay_2018_19.csv',format="csv", sep=",", inferSchema="true", header="true")
pbpDF = pbpDF.union(spark.read.load('playByPlay_2017_18.csv',format="csv", sep=",", inferSchema="true", header="true"))
pbpDF = pbpDF.union(spark.read.load('playByPlay_2016_17.csv',format="csv", sep=",", inferSchema="true", header="true"))
pbpDF = pbpDF.union(spark.read.load('playByPlay_2015_16.csv',format="csv", sep=",", inferSchema="true", header="true"))

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








#%% 







playerList = pbpDF.select('PLAYER1_NAME', 'PLAYER1_ID', 'GAME_ID').distinct().collect()

def getPlayerID(playerName):
    return [row.PLAYER1_ID for row in playerList if row.PLAYER1_NAME == playerName][0]

def generatePlayerData(player, returnFormat, verbose):

    ## player = playerList[i].PLAYER1_NAME
    ## playerID = playerList[i].PLAYER1_ID
    ## gameID = playerList[i].GAME_ID
    
    playerID = getPlayerID(player)

    if player == None or playerID == None:
        return None

    gameID = [row.GAME_ID for row in playerList if row.PLAYER1_NAME == player]

    #%%
    ## For each player and each game
    #playerGame = pbpDF.filter(pbpDF.GAME_ID == gameID)

    playerGame = pbpDF.filter(pbpDF.GAME_ID.isin(gameID))
    playerGame.orderBy(["GAME_ID","TIME"], ascending = [1,1])

    playerGame = playerGame.withColumn('IsOnFloor', f.when(array_contains(playerGame['PlayersOnFloor'], playerID), True).otherwise(False))

    my_window = Window.partitionBy().orderBy('GAME_ID', 'TIME')
    playerGame = playerGame.withColumn('PosDuration', f.lead(playerGame.TIME).over(my_window) - playerGame.TIME)
    playerGame = playerGame.withColumn('PosDuration', f.when(playerGame.PosDuration < 0, 0).otherwise(playerGame.PosDuration))
    playerGame = playerGame.withColumn('RestDuration', f.when(playerGame.IsOnFloor == True, 0).otherwise(playerGame.PosDuration))
    playerGame = playerGame.withColumn('PlayDuration', f.when(playerGame.IsOnFloor == False, 0).otherwise(playerGame.PosDuration))

    playerGame = playerGame.withColumn('TotalRest', f.sum('RestDuration').over(Window.orderBy(f.col('TIME').asc()).rowsBetween(Window.unboundedPreceding, 0)))
    playerGame = playerGame.withColumn('TotalPlay', f.sum('PlayDuration').over(Window.orderBy(f.col('TIME').asc()).rowsBetween(Window.unboundedPreceding, 0)))
    #playerGame = playerGame.withColumn('NetPlay', playerGame.TotalPlay - playerGame.TotalRest)
    playerGame = playerGame.withColumn('R2PRatio', f.when(playerGame.TotalPlay == 0, 0).otherwise(playerGame.TotalRest / playerGame.TotalPlay))

    ## Stats Available: Assists, Blocks, ORB, DRB, FGA, Points Scored (FT and FG), FTA, Steal, Turnover, 3PM
    playerGame = playerGame.withColumn('AstAdded', f.when(playerGame.ASSIST_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('BlockAdded', f.when(playerGame.BLOCK_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('ORBAdded', f.when(playerGame.REBOUND_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('DRBAdded', f.when(playerGame.REBOUND_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('FGAAdded', f.when(playerGame.SHOT_PLAYER_ID == playerID, 1).otherwise(0))
    ## Extra work required to get points scored
    my_window2 = Window.partitionBy().orderBy('GAME_ID', 'TIME').rowsBetween(-sys.maxsize, 0)
    playerGame = playerGame.withColumn('FGPoints', f.when((playerGame.SHOT_PLAYER_ID == playerID) & (playerGame.SHOT_MADE == True), playerGame.TOTAL_POINTS_SCORED).otherwise(lit(None)))  
    playerGame = playerGame.withColumn('FGPoints', f.last('FGPoints', True).over(Window.partitionBy('GAME_ID').orderBy('TIME').rowsBetween(-sys.maxsize, 0)))
    playerGame = playerGame.withColumn('FGPoints', f.when(playerGame.FGPoints.isNull(), 0).otherwise(playerGame.FGPoints))  
    playerGame = playerGame.withColumn('FGPtsAdded', playerGame.FGPoints - f.lag(playerGame.FGPoints).over(my_window))
    playerGame = playerGame.withColumn('FTPtsAdded', f.when((playerGame.FREE_THROW_PLAYER_ID == playerID) & (playerGame.FREE_THROW_MADE == True), 1).otherwise(0))
    playerGame = playerGame.withColumn('TotalPtsAdded', playerGame.FTPtsAdded + playerGame.FGPtsAdded)
    playerGame = playerGame.withColumn('TotalPtsAdded', f.when(playerGame.TotalPtsAdded < 0, 0).otherwise(playerGame.TotalPtsAdded))
    playerGame = playerGame.withColumn('TotalPtsAdded', f.when(playerGame.TotalPtsAdded > 3, 0).otherwise(playerGame.TotalPtsAdded))

    ## Got Total Points Added, now getting the rest
    playerGame = playerGame.withColumn('FTAAdded', f.when((playerGame.FREE_THROW_PLAYER_ID == playerID), 1).otherwise(0))
    playerGame = playerGame.withColumn('StealAdded', f.when(playerGame.STEAL_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('TOAdded', f.when(playerGame.TURNOVER_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('ThreePMAdded', f.when((playerGame.SHOT_PLAYER_ID == playerID) & (playerGame.TotalPtsAdded == 3), 1).otherwise(0))
    playerGame = playerGame.withColumn('PFAdded', f.when((playerGame.FOULED_BY_PLAYER_ID == playerID), 1).otherwise(0))
    
    ## Is it a close game? 
    #playerGame = playerGame.withColumn('Close', f.when(((playerGame.SCOREMARGIN < 10) & (playerGame.SCOREMARGIN > (0-10))), 1.05).otherwise(1))
    #playerGame = playerGame.withColumn('VeryClose', f.when(((playerGame.SCOREMARGIN < 5) & (playerGame.SCOREMARGIN > (0-5))), 1.05).otherwise(1))
    playerGame = playerGame.withColumn('ScoreMultiplier', f.when(playerGame.SCOREMARGIN.isNotNull(), (1 / playerGame.SCOREMARGIN) + 1).otherwise(1))

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
                            (playerGame.FGAAdded * -0.67) + (playerGame.FTAAdded * -0.2945), 5) * (playerGame.ScoreMultiplier))


    #playerGameCSV = playerGame

    #playerGameCSV = playerGame.select('GAME_ID', 'TotalRest', 'TotalPlay', 'R2PRatio', 'BPMAdded')
    playerGame = playerGame.withColumn('PlayerName', lit(player))
    playerGame = playerGame.withColumn('PlayerID', lit(playerID))

    playerGame = playerGame.filter(playerGame.BPMAdded != 0)    
    pdf = playerGame.toPandas()
    
    if verbose: 
        print('Player Name :', player)
        print('Total Data Points : ', playerGame.count()) 
        print('Number of Games : ', len(gameID))

    if returnFormat == 'pandas':
        return pdf
              
    else: 
        return playerGame
    







#%%





#generatePlayerData('Joel Embiid', 'spark', True)
pdf = generatePlayerData('JR Smith', 'pandas', True)





#%%
#%matplot plt
#pdf.plot(x='R2PRatio', y='BPMAdded', style='o')
#plt.show()





sns.lmplot('R2PRatio','BPMAdded', pdf)




#%%

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols = ['TotalRest', 'TotalPlay'], outputCol = 'features')
vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['features', 'MV'])
vhouse_df.show(3)













#%%
'''

## Get all the players (Team 1) 
## TO DO --> Do this for more than just the Player1_Name column
playerList = pbpDF.select('PLAYER1_NAME', 'PLAYER1_ID', 'GAME_ID').distinct().collect()
firstRun = True
#for i in tqdm(range(1, len(playerList))):
for i in tqdm(range(22, 23)):

    player = playerList[i].PLAYER1_NAME
    playerID = playerList[i].PLAYER1_ID
    ## gameID = playerList[i].GAME_ID

    if player == None or playerID == None:
        continue

    gameID = [row.GAME_ID for row in playerList if row.PLAYER1_NAME == player]

    #%%
    ## For each player and each game
    #playerGame = pbpDF.filter(pbpDF.GAME_ID == gameID)

    playerGame = pbpDF.filter(pbpDF.GAME_ID.isin(gameID))
    playerGame.orderBy(["GAME_ID","TIME"], ascending = [1,1])

    playerGame = playerGame.withColumn('IsOnFloor', f.when(array_contains(playerGame['PlayersOnFloor'], playerID), True).otherwise(False))

    my_window = Window.partitionBy().orderBy('GAME_ID', 'TIME')
    playerGame = playerGame.withColumn('PosDuration', f.lead(playerGame.TIME).over(my_window) - playerGame.TIME)
    playerGame = playerGame.withColumn('RestDuration', f.when(playerGame.IsOnFloor == True, 0).otherwise(playerGame.PosDuration))
    playerGame = playerGame.withColumn('PlayDuration', f.when(playerGame.IsOnFloor == False, 0).otherwise(playerGame.PosDuration))

    playerGame = playerGame.withColumn('TotalRest', f.sum('RestDuration').over(Window.orderBy(f.col('TIME').asc()).rowsBetween(Window.unboundedPreceding, 0)))
    playerGame = playerGame.withColumn('TotalPlay', f.sum('PlayDuration').over(Window.orderBy(f.col('TIME').asc()).rowsBetween(Window.unboundedPreceding, 0)))
    #playerGame = playerGame.withColumn('NetPlay', playerGame.TotalPlay - playerGame.TotalRest)
    playerGame = playerGame.withColumn('R2PRatio', f.when(playerGame.TotalPlay == 0, 0).otherwise(playerGame.TotalRest / playerGame.TotalPlay))

    ## Stats Available: Assists, Blocks, ORB, DRB, FGA, Points Scored (FT and FG), FTA, Steal, Turnover, 3PM
    playerGame = playerGame.withColumn('AstAdded', f.when(playerGame.ASSIST_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('BlockAdded', f.when(playerGame.BLOCK_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('ORBAdded', f.when(playerGame.REBOUND_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('DRBAdded', f.when(playerGame.REBOUND_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('FGAAdded', f.when(playerGame.SHOT_PLAYER_ID == playerID, 1).otherwise(0))
    ## Extra work required to get points scored
    my_window2 = Window.partitionBy().orderBy('GAME_ID', 'TIME').rowsBetween(-sys.maxsize, 0)
    playerGame = playerGame.withColumn('FGPoints', f.when((playerGame.SHOT_PLAYER_ID == playerID) & (playerGame.SHOT_MADE == True), playerGame.TOTAL_POINTS_SCORED).otherwise(lit(None)))  
    playerGame = playerGame.withColumn('FGPoints', f.last('FGPoints', True).over(Window.partitionBy('GAME_ID').orderBy('TIME').rowsBetween(-sys.maxsize, 0)))
    playerGame = playerGame.withColumn('FGPoints', f.when(playerGame.FGPoints.isNull(), 0).otherwise(playerGame.FGPoints))  
    playerGame = playerGame.withColumn('FGPtsAdded', playerGame.FGPoints - f.lag(playerGame.FGPoints).over(my_window))
    playerGame = playerGame.withColumn('FTPtsAdded', f.when((playerGame.FREE_THROW_PLAYER_ID == playerID) & (playerGame.FREE_THROW_MADE == True), 1).otherwise(0))
    playerGame = playerGame.withColumn('TotalPtsAdded', playerGame.FTPtsAdded + playerGame.FGPtsAdded)
    playerGame = playerGame.withColumn('TotalPtsAdded', f.when(playerGame.TotalPtsAdded < 0, 0).otherwise(playerGame.TotalPtsAdded))
    playerGame = playerGame.withColumn('TotalPtsAdded', f.when(playerGame.TotalPtsAdded > 3, 0).otherwise(playerGame.TotalPtsAdded))

    ## Got Total Points Added, now getting the rest
    playerGame = playerGame.withColumn('FTAAdded', f.when((playerGame.FREE_THROW_PLAYER_ID == playerID), 1).otherwise(0))
    playerGame = playerGame.withColumn('StealAdded', f.when(playerGame.STEAL_PLAYER_ID == playerID, 1).otherwise(0))
    playerGame = playerGame.withColumn('TOAdded', f.when(playerGame.TURNOVER_PLAYER_ID == playerID, 1).otherwise(0))
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


    #playerGameCSV = playerGame

    #playerGameCSV = playerGame.select('GAME_ID', 'TotalRest', 'TotalPlay', 'R2PRatio', 'BPMAdded')
    playerGame = playerGame.withColumn('PlayerName', lit(player))
    playerGame = playerGame.withColumn('PlayerID', lit(playerID))

    playerGame = playerGame.filter(playerGame.BPMAdded != 0)    
    pdf = playerGame.toPandas()

    print(player, playerGame.count()) 


    if firstRun: 

        collectorDF = playerGame

        #playerGameCSV.repartition(1).write.csv(path="regressionData.csv", mode="append", header = True)
        
        #playerGameCSV.coalesce(1).write.csv('regressionData.csv', header = True)
        #playerGameCSV.write.csv('regressionData.csv')
        firstRun = False
    else:

        collectorDF = collectorDF.union(playerGameCSV)

        #playerGameCSV.repartition(1).write.csv(path="regressionData.csv", mode="append")
        #playerGameCSV.coalesce(1).write.csv('regressionData.csv', mode = 'append', header = True)
        #playerGameCSV.write.save(path='regressionData.csv', format='csv', mode='append', sep=',')


# %%



## collectorDF.coalesce(1).write.csv('regressionData.csv', header = True)

# %%
'''