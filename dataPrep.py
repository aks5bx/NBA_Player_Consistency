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
import statistics
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sc = SparkContext.getOrCreate() ## optionally takes in a spark conf as a parameter
spark = SparkSession(sc)

#pbpDF = pd.read_csv('playByPlay_2018_19.csv')
#playerList = pbpDF[['PLAYER1_NAME', 'PLAYER1_ID', 'GAME_ID']].drop_duplicates()







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
    playerGame = playerGame.withColumn('SCOREMARGIN', f.when(playerGame.SCOREMARGIN == 'TIE', 0).otherwise(playerGame.SCOREMARGIN))
    playerGame = playerGame.withColumn('SCOREMARGIN', f.last('SCOREMARGIN', True).over(Window.partitionBy('GAME_ID').orderBy('TIME').rowsBetween(-sys.maxsize, 0)))
    playerGame = playerGame.withColumn('ScoreMultiplier', f.when(playerGame.SCOREMARGIN.isNotNull(), (10 / playerGame.SCOREMARGIN)).otherwise(1))

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
    
    if verbose: 
        print('Player Name :', player)
        print('Total Data Points : ', playerGame.count()) 
        print('Number of Games : ', len(gameID))

    print('DATA GENERATED')

    if returnFormat == 'pandas':
        return pdf
              
    else: 
        playerGame = playerGame.select('GAME_ID', 'R2PRatio', 'BPMAdded', 'TIME')
        return playerGame
    







#%%





#generatePlayerData('Joel Embiid', 'spark', True)
# pdf = generatePlayerData('Jeff Teague', 'pandas', False)
#playerGame = generatePlayerData('Reggie Jackson', 'spark', False)

## SPECIFIC GAME R2P RATIO
#games = list(set(pdf['GAME_ID']))
#pdfScore = pdf[(pdf.GAME_ID == games[21])]
#pdfScore = pdfScore[np.abs(pdfScore.BPMAdded - pdfScore.BPMAdded.mean())<=(2*pdfScore.BPMAdded.std())]
#sns.regplot('R2PRatio','BPMAdded', pdfScore, robust = True)

## SCORE MARGIN
#games = list(set(pdf['GAME_ID']))
#pdfScore = pdf[(~pdf.SCOREMARGIN.isna()) & (pdf.GAME_ID == games[6])]
#pdfScore['SCOREMARGIN'] = pdfScore['SCOREMARGIN'].astype('int')
#sns.regplot('SCOREMARGIN','BPMAdded', pdfScore, robust = False)


############
### IDEA ###
############

# Understand each player's performance clusters in order to optimize lineups
## You want a lineup where you have players who are different, unique clusters
## Look at Jokic for good cluster example
## Look at clusters on a game by game level
## List out all of the clusters a player has over all of their games
## Aggregate all of the clusters (remove outliers, keep only the core identity) 
## to form the general performance clusters of a player



#%%
#%matplot plt
#pdf.plot(x='R2PRatio', y='BPMAdded', style='o')
#plt.show()


#sns.lmplot('R2PRatio','BPMAdded', pdf)




#%%
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def generateClusters(playerGame, init, gameNumber):

    gameList = playerGame.select("GAME_ID").distinct().collect()
    gameList = [game[0] for game in gameList]

    if init:
        playerGameSingle = playerGame.filter(playerGame.GAME_ID.isin(gameList[0:30]))
    
    else: 
        playerGameSingle = playerGame.filter(playerGame.GAME_ID.isin(gameList[gameNumber]))


    vectorAssembler = VectorAssembler(inputCols = ['R2PRatio', 'BPMAdded'], outputCol = 'features')
    VplayerGame = vectorAssembler.transform(playerGameSingle)
    VplayerGame = VplayerGame.select('features')

    kmeans = KMeans().setK(3).setSeed(1)
    model = kmeans.fit(VplayerGame)

    predictions = model.transform(VplayerGame)

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)

    rows = model.transform(VplayerGame).select('prediction').collect()
    
    
    if init:
        print("Silhouette with squared euclidean distance = " + str(silhouette))

    # coreCenters.append(model.clusterCenters())

    #core0Count = len([1 for row in rows if row[0] == 0])
    #ore1Count = len([1 for row in rows if row[0] == 1])
    #core2Count = len([1 for row in rows if row[0] == 2])
    #ore3Count = len([1 for row in rows if row[0] == 3])

    #coreCenters.extend([model.clusterCenters()[0] for i in range(core0Count)])
    #coreCenters.extend([model.clusterCenters()[1] for i in range(core1Count)])
    #coreCenters.extend([model.clusterCenters()[2] for i in range(core2Count)])
    #coreCenters.extend([model.clusterCenters()[3] for i in range(core3Count)])
    
    #if core0Count == max([core0Count, core1Count, core2Count, core3Count]):
    #    coreCenters.append(model.clusterCenters()[0])
    #elif core1Count == max([core0Count, core1Count, core2Count, core3Count]):
    #    coreCenters.append(model.clusterCenters()[1])
    #elif core2Count == max([core0Count, core1Count, core2Count, core3Count]):
    #    coreCenters.append(model.clusterCenters()[2])
    #elif core3Count == max([core0Count, core1Count, core2Count, core3Count]):
    #    coreCenters.append(model.clusterCenters()[3])

    return model.clusterCenters()


def manhattanDistance(coor1, coor2):
    x1 = coor1[0]
    y1 = coor1[1]

    x2 = coor2[0]
    y2 = coor2[1]

    dist = abs(x2 - x1) + abs(y2 - y1)

    return dist

def clusterDifference(anchorClusters, newClusters): 
    anchorCluster1 = anchorClusters[0]
    anchorCluster2 = anchorClusters[1]
    anchorCluster3 = anchorClusters[2]

    newClusters1 = newClusters[0]
    newClusters2 = newClusters[1]
    newClusters3 = newClusters[2]


    distance1 =  manhattanDistance(anchorCluster1, newClusters1) + manhattanDistance(anchorCluster2, newClusters2) + manhattanDistance(anchorCluster3, newClusters3)

    distance2 =  manhattanDistance(anchorCluster1, newClusters1) + manhattanDistance(anchorCluster2, newClusters3) + manhattanDistance(anchorCluster3, newClusters2)

    distance3 =  manhattanDistance(anchorCluster1, newClusters2) + manhattanDistance(anchorCluster2, newClusters1) + manhattanDistance(anchorCluster3, newClusters3)

    distance4 =  manhattanDistance(anchorCluster1, newClusters2) + manhattanDistance(anchorCluster2, newClusters3) + manhattanDistance(anchorCluster3, newClusters1)

    distance5 =  manhattanDistance(anchorCluster1, newClusters3) + manhattanDistance(anchorCluster2, newClusters1) + manhattanDistance(anchorCluster3, newClusters2)

    distance6 =  manhattanDistance(anchorCluster1, newClusters3) + manhattanDistance(anchorCluster2, newClusters2) + manhattanDistance(anchorCluster3, newClusters1)

    minDist = min([distance1, distance2, distance3, distance4, distance5, distance6])


    return minDist 


def genPlayerVariance(player):
    playerGame = generatePlayerData(player, 'spark', False)

    anchorClusters = generateClusters(playerGame, True, None)

    totalDiff = []
    for gameNumber in tqdm(range(31, 61)):
        gameClusters = generateClusters(playerGame, False, gameNumber)
        totalDiff.append(clusterDifference(anchorClusters, gameClusters))

    return statistics.mean(totalDiff)


results = []
playerListLoop = ['LeBron James', 'Nikola Jokic', 'Kyrie Irving', 'Kyle Lowry', 'Anthony Davis', 'Damian Lillard', 'Lou Williams', 'Chris Paul', 'John Wall', 'Bradley Beal', 'Jimmy Butler']

for player in tqdm(playerListLoop):
    results.append(genPlayerVariance(player))




#x = []
#y = []
#for core in coreCenters: 
#    x.append(core[0])
#    y.append(core[1])

#plt.scatter(x, y)

#splits = VplayerGame.randomSplit([0.7, 0.3])
#train_df = splits[0]
#test_df = splits[1]

#cost = np.zeros(20)
#for k in range(2,20):
 #   kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
  #  model = kmeans.fit(VplayerGame.sample(False,0.1, seed=42))
   # cost[k] = model.computeCost(VplayerGame) # requires Spark 2.0 or lat

#fig, ax = plt.subplots(1,1, figsize =(8,6))
#ax.plot(range(2,20),cost[2:20])
#ax.set_xlabel('k')
#ax.set_ylabel('cost')

#lr = LinearRegression(featuresCol = 'features', labelCol='BPMAdded', maxIter=10, regParam=0.3, elasticNetParam=0.8)
#lr_model = lr.fit(train_df)
#rint("Coefficients: " + str(lr_model.coefficients))
#print("Intercept: " + str(lr_model.intercept))

#trainingSummary = lr_model.summary
#print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
#print("r2: %f" % trainingSummary.r2)








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