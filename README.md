# ESG Investing II
 
This repository contains Python code to simulate the performance of portfolios using different investment strategies. Specifically, it focuses on comparing the performance of traditional investment strategies with those incorporating Environmental, Social, and Governance (ESG) screening.

The code uses historical stock price data and ESG scores to create an investment universe of stocks. Then, it applies various investment strategies to this universe, simulates their performance over a specified period, and compares their performance using several key metrics such as cumulative returns, daily volatility, value at risk, conditional value at risk, and drawdown.

Overview
The project is structured around two main classes:

stock_universe: This class represents the investment universe of stocks. It includes methods for fetching historical stock price and ESG score data, creating the stock universe, and applying ESG screening to select stocks.

Portfolio_Simulation: This class is used to simulate and analyze the performance of portfolios using different investment strategies. It includes methods for computing portfolio metrics, running the simulation, plotting results, and performing statistical tests to compare the performance of different strategies.

The simulation results are plotted using a joyplot for each performance metric. Additionally, the Mann-Whitney U test is performed to statistically compare the performance of different investment strategies.
