from os.path import curdir, join
import pandas as pd
import matplotlib.pyplot as plt


from dinosaur.research.analysis import IndustryCycling
from dinosaur.backtesting.strategy_lib import IndustryCyclingStrategy  # FundSignalTesting,


def run():
    """

    :return:
    """
    data_path = join(curdir, 'data')
    # Initialize Analyzing module
    research = IndustryCycling(data_path)
    # Calculate the estimated industrial allocations for each fund, in monthly step, and save to local
    # This step takes about 5 minutes with 4 cores multi-processing
    research.get_fund_allocation_est(obs_months=1, multi=True)
    # Calculate neutralized allocations signals in window of 12 months, and save to local
    # This step takes about 1.5 minutes
    neutralized = research.get_fund_neutralized_allocation(obs_months=1, neutralized_months=12)  # , renew=True)
    # Calculate the t-Stats of the rank_IC for the neutralized signals as the final fund selection indicator
    fund_cycling_signals = research.cal_signals(neutralized)
    # Calculate rank_ICs with respect to different lags
    ic_holder, ic_mean_holder = research.rank_ic_cal(fund_cycling_signals)
    # Plot IC
    research.plt_bar_series(pd.Series(ic_mean_holder))
    research.ic_analysis(ic_holder['1_lags'], '1_lags', True)
    research.ic_analysis(ic_holder['3_lags'], '3_lags', True)
    research.ic_analysis(ic_holder['11_lags'], '11_lags', True)
    # Formulate a simple back_testing on funds based on the selection indicators, splitting into 5 groups
    research.simple_back_testing(fund_cycling_signals)

    # Next, constructing the industry cycling strategy
    # Refer to the report, quarterly industry allocations estimations are need, so we run the allocation
    # estimation with different parameters and save them for future usage.
    research.get_fund_allocation_est(obs_months=3, multi=True)
    # Now we can run back_testing based on the just saved allocations and previously calculated fund
    # selection indicators
    strategy = IndustryCyclingStrategy(data_path)
    strategy.run('2011-12-30', '2018-05-30')
    strategy.summary.to_csv(join(curdir, 'intermediate_results', 'CyclingStrategy.csv'))
    print(strategy.summary)
    strategy.nav.plot()
    plt.savefig(join(curdir, 'intermediate_results', 'CyclingStrategyNAV.png'))

    # plt.show()


if __name__ == '__main__':
    run()
