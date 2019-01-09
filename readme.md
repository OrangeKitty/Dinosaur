# IntroSpec of Industry Cycling Research Project

## Package Structure Illustration

* **mod**
    * **brokerage**: broker elements for backtesting, like account, order book, etc.
    * **environment**: backtesting data module
    * **interface**: environment api and strategy api
    
* **test**
    * **strategy_lib**: example strategy
* utils
    * **indicator_lib**: commonly used indicators to evaluate portfolios or financial instruments
    * **stats_lib**: integrated all most usually used indicators
    * **timer**: tools to log processing time, including function timer decorator, progress bar, etc. 