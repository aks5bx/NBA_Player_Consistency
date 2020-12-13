# NBA Player Consistency

## Summary
The challenge this project looks to address is quantifying how consistent NBA players are from game to game. Specifically, this project looks to account for in-game variation. Say a player has game (Game 1) where they score 30 points, all 30 of which come in the last quarter. Then let’s say a player has a game (Game 2) where they score 30 points, with 10 coming in quarter 1, 10 coming in quarter 2, 10 coming in quarter 3, and none coming in quarter four. Conventionally, one might say that the player scored 30 points in both games and therefore that player is consistent. However, this project detects the inconsistency and reports it given the two games were very different in nature. This project also uses a variation of Adjusted BPM in order to drive the calculations.  

- Powerpoint presentation in the repo has detailed explanation of the process with diagrams, examples, and step by step walk through

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
