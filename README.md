# NBA_InGame_Rest

## Goal: quantify the consistency of a player’s performance in a game or stretch of games

## Steps Taken 
- 1: Generate Plot of Performance over course of many games (Baseline Plot)
- 2: Create a performance plot for games of interest (Game Plot)
- 3: Quantify the difference between the Game Plot and the Baseline Plot

The idea here is that the more a player deviates from their expected performance (defined by the Baseline Plot) the less consistent their performance is. 

## Metrics Used: 
- Rest to Play Ratio (R2PRatio): Proxy metric to define time elapsed or the flow of the game, personalized to the player of interest 
- BPM Added: Simplified version of Box Plus Minus, defined for each individual player on a per-possession level 

## Example Baseline Plot (please excuse grainy quality!)
![alt text](https://github.com/aks5bx/NBA_InGame_Rest/blob/develop/BaselinePlot.png?raw=true)

## Example Game Plot (please excuse grainy quality!)
![alt text](https://github.com/aks5bx/NBA_InGame_Rest/blob/develop/GamePlot.png?raw=true)

## Technology Used 
- Apache Spark, Spark ML Lib 
- Python + Pandas 

## Methods Applied 
- K-Means Clustering
