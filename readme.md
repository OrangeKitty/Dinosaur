# IntroSpec of Industry Cycling Research Project

## How to Run
### 1. In Terminal
\> python main.py

Intermediate results will be logged in the ./intermediate_results folder


### 2. By Jupyter NoteBook

The processing framework has been built in the ./demo.ipynb. Just go.

*(Multiprocessing module can't be used in Jupyter Notebook.
Some calculations may take more than that in terminal running.)*


## Package Structure Illustration
* **backtesting**
    * **brokerage**: broker elements for backtesting, like account, order book, etc.
    * **environment**: backtesting data module
    * **interface**: environment api and strategy api
    * **strategy_lib**: example strategy and research testing strategy
* **data**: placeholder for original data         
* **intermediate_results**: placeholder for intermediate results
* **research**
    * **analysis**: the main analyzing module        
* utils
    * indicator_lib: commonly used indicators to evaluate portfolios or financial instruments
    * stats_lib: integrated all most usually used indicators
    * timer: tools to log processing time, including function timer decorator, progress bar, etc. 