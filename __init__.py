import pandas as pd
import os
import numpy as np
import argparse
 
os.chdir('/home/mis/MonteCarloAnalysis')
os.system('cls || clear')
 
class MonteCarloAnalysis:
    def __init__(self, initial_value = 300000, margin_call = 80000, slippage = 0, commission = 0, P='profit', C='contracts'):
        self.journals = [os.path.join('data', j) for j in os.listdir('data') if '.csv' in j]
        self.initial_value = initial_value
        self.margin_call = margin_call
        self.slippage = slippage
        self.commission = commission
        self.profit_column = P
        self.contract_column = C
        self.merge_trades()
 
    def merge_trades(self):
        self.trade_df = pd.DataFrame()
        for j in self.journals:
            j_df = pd.read_csv(j)
            self.trade_df = self.trade_df.append(j_df[j_df[self.profit_column]!=0], ignore_index=True)
        self.trade_df.index = range(self.trade_df.shape[0])
 
    def sample(self, yearly_trades):
        return self.trade_df[[self.profit_column, self.contract_column]].sample(yearly_trades)
 
    def compose(self, yearly_trades, sample_n):
        self.yearly_trades = yearly_trades
        self.sample_n = sample_n
 
        self.equity_curves = np.zeros([sample_n, yearly_trades])
        for s in range(sample_n):
            sampling_df = self.sample(yearly_trades)
            pnl = (sampling_df[self.profit_column] * 200 - sampling_df[self.contract_column].abs() * (self.slippage + self.commission)).cumsum() + self.initial_value
            self.equity_curves[s] = pnl
 
    def risk_of_ruin(self):
        try:
            getattr(self, "perf_risk_of_ruin")
 
        except:
            count = 0
            for m in range(self.sample_n):
                for n in range(self.yearly_trades):
                    if self.equity_curves[m, n] < self.margin_call:
                        count += 1
                        break
            self.perf_risk_of_ruin = count/self.sample_n
 
        return self.perf_risk_of_ruin
 
    def median_return(self):
        try:
            getattr(self, "perf_median_ret")
 
        except:
            ret_array = np.zeros(self.sample_n)
            for m in range(self.sample_n):
                ret_array[m] = self.equity_curves[m, -1]/self.initial_value - 1
 
            self.perf_median_ret = np.median(ret_array)
 
        return self.perf_median_ret
 
    def prob(self):
        try:
            getattr(self, "perf_prob")
 
        except:
            count = 0
            for m in range(self.sample_n):
                if self.equity_curves[m, -1] > self.initial_value:
                    count += 1
 
            self.perf_prob = count / self.sample_n
 
        return self.perf_prob
 
    def drawdown(self):
        try:
            getattr(self, "perf_median_dd")
 
        except:
            mdd_array = np.array([])
            for m in range(self.sample_n):
                equity_df = pd.DataFrame(self.equity_curves[m])
                roll_max = equity_df.rolling(100, min_periods=1).max()
                daily_drawdown = equity_df/roll_max - 1.0
                max_daily_drawdown = daily_drawdown.rolling(100, min_periods=1).min()
                mdd_array = np.append(mdd_array, max_daily_drawdown)
 
            self.perf_median_dd = np.median(mdd_array) * -1
 
        return self.perf_median_dd
 
    def return_dd(self):
        try:
            getattr(self, "perf_return_dd")
 
        except:
            self.perf_return_dd = self.perf_median_ret / self.perf_median_dd
 
        return self.perf_return_dd
 
    def performance(self):
        print("\n\n\tMonte Carlo Analysis\n\n")
        print("\tPerformance\n")
 
        print("\tRisk of Ruin: {:.2%}".format(self.risk_of_ruin()))
        print("\tMedian Maximum Drawdown: {:.2%}".format(self.drawdown()))
        print("\tMedian Return: {:.2%}".format(self.median_return()))
        print("\tReturn/Drawdown: {:.2%}".format(self.return_dd()))
        print("\tProb > 0: {:.2%}".format(self.prob()))
 
        print('\n\n\tThreshold\n')
        print('\tRisk of Ruin: < 10%')
        print('\tMedian Maximum Drawdown: < 40%')
        print('\tMedian Return: > 40%')
        print('\tReturn/Drawdown: > 2.0\n\n')
 
    def performance_test(self):
       
        try:
            getattr(self, "perf_risk_of_ruin")
            getattr(self, "perf_median_dd")
            getattr(self, "perf_median_ret")
            getattr(self, "perf_return_dd")
            getattr(self, "perf_prob")
        except:
            self.performance()
 
        print("\tTest\n")
        if self.perf_risk_of_ruin < 0.1:
            print('\tRisk of Ruin: PASS')
        else:
            print('\tRisk of Ruin: FAIL')
 
        if self.perf_median_dd < 0.4:
            print('\tMedian Maximum Drawdown: PASS')
        else:
            print('\tMedian Maximum Drawdown: FAIL')
 
        if self.perf_median_ret > 0.4:
            print('\tMedian Return: PASS')
        else:
            print('\tMedian Return: FAIL')
 
        if self.perf_return_dd > 2:
            print('\tReturn/Drawdown: PASS\n\n')
        else:
            print('\tReturn/Drawdown: FAIL\n\n')
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-value", '-IV', help='Initial value for portfolio', type=int, default=300000)
    parser.add_argument("--margin", "-MG", help='Threshold of margin call', type=int, default=80000)
    parser.add_argument("--yearly-trade", '-YT', help='# of trades per year', type=int, default=100)
    parser.add_argument("--samples", '-SP', help='# of samples', type=int, default=30)
    parser.add_argument("--slippage", '-SL', help='Slippage cost', type=int, default=0)
    parser.add_argument("--commission", "-CM", help='Commission cost', type=int, default=0)
    parser.add_argument("--profit-column", "-PC", help='Profit column name', default='profit')
    parser.add_argument("--contract-column", "-CC", help='Contract column name', default='contracts')
 
    args = parser.parse_args()
    m = MonteCarloAnalysis(args.initial_value, args.margin, args.slippage, args.commission, args.profit_column, args.contract_column)
    m.compose(args.yearly_trade, args.samples)
   
    # m.performance()
    m.performance_test()
