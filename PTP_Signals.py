
# coding: utf-8

# # Define Constants

# In[1]:

import pandas as pd
import numpy as np
import talib
from pandas import Series,DataFrame
import datetime
import itertools
import sys



WEEK_PERIOD = 5
MONTH_PERIOD = 20
MAX_RETURN_VALUE = 10
MACD_RSI_WINDOW = 10
START_DATE = datetime.date(2011,1,1)
    
NUMBER_OF_RANDOM_SAMPLES= 300
CONFIDENCE_INTERVAL = 0.95
TD_SETUP_BARS_TO_LOOK_BACK = 4 # Look back # of  periods to see if the condition is completed
TD_BARS_FOR_SETUP = 9 # Look at n consecutive periods to see if there's a TDSETUP
TD_SETUP_B = 2
TD_BARS_FOR_COUNTDOWN = 13
TD_COUNTDOWN_BARS_TO_LOOK_BACK = 2
TD_COUNTDOWN_INCOMPLETE_BAR_SYMBOL = 130
TD_COUNTDOWN_BAR8 = 8
TD_COUNTDOWN_BAR5 = 5
PERIODS = ['1day', '2day', '1wk', '2wk', '1mo']


RSI_DOUBLE_WINDOW = 10
RSI_MA_WINDOW = 35
RSI_BUY_LEVEL = 45 #Normally 30
RSI_SELL_LEVEL = 55 #Normally 70
RSI_CALC_WINDOW = 50 #No. of lookback days for RSI calculation (normally use 14day)


# # Functions

# In[2]:

def demark(in_df) :
    stock_df = in_df[pd.notnull(in_df['adj_close']) &
                     pd.notnull(in_df['adj_high']) &
                     pd.notnull(in_df['adj_low'])].copy()
    print('Computing DeMark, RSI and MACD for ', (stock_df.bb_ticker.unique()), len(stock_df))
    sys.stdout.flush()
    
    def find_price_flip(df): 

        result = pd.DataFrame(index = df.index)

        # compute bullish_price_flip EDIT BEARISH NOW
        result['price_flip']  = -1*((df.adj_close < df.adj_close.shift(TD_SETUP_BARS_TO_LOOK_BACK)) &
                   (df.adj_close.shift(1) > df.adj_close.shift(TD_SETUP_BARS_TO_LOOK_BACK+1)) ).astype(float)

        result['price_flip'].replace(to_replace=np.NaN, value=0, inplace=True) #replace NaNs, so we can convert to int

        return(result['price_flip'].values.astype(int))

    def find_td_setup(df, bars_to_look_back=TD_SETUP_BARS_TO_LOOK_BACK, bars_to_setup=TD_BARS_FOR_SETUP):
        result = df.copy()

        result['adj_close_gt_look_back'] = (df.adj_close > df.adj_close.shift(bars_to_look_back)).replace(to_replace=np.NaN, value=0).astype(int)
        result['adj_close_lt_look_back'] = (df.adj_close < result.adj_close.shift(bars_to_look_back)).replace(to_replace=np.NaN, value=0).astype(int)

        result['td_setup'] = 0
        for i in range(1,bars_to_setup+1):
            result['bearish_sum'] = -(pd.Series.rolling(result['adj_close_lt_look_back'], window=i, min_periods=1).sum())
            
            result['td_setup'] = result['bearish_sum'].where((result['bearish_sum'] == -i), result['td_setup'])
            
        result['td_setup'] = result['td_setup'].where((result['td_setup'].shift(1) != -bars_to_setup), 0)

        #result['td_setup'] = result['bearish_sum'].where((result['bearish_sum'] == i), result['td_setup'])

        #result['td_setup'] = result['bearish_series_'+str(i)].where((result['bearish_series_'+str(i)] == -i), result['td_setup'])

        # remove consecutive bar9s from series
        #result['td_setup'] = result['td_setup'].where((result['td_setup'].shift(1) != bars_to_setup), 0)


        return(result['td_setup'].astype(int))

    def find_setup_start(df, bars_to_look_back=TD_SETUP_BARS_TO_LOOK_BACK, bars_to_setup=TD_BARS_FOR_SETUP):
        result = pd.DataFrame(index=df.index)

        result['setup_start'] = ((df['td_setup'].abs() == 1) &
                                    (df['td_setup'].shift(-(bars_to_setup-1)).abs() == bars_to_setup)).replace(to_replace=np.NaN, value=0).astype(int)
        return(result['setup_start'].values.astype(int))

    def number_td_setups(df, bars_to_look_back=TD_SETUP_BARS_TO_LOOK_BACK, bars_to_setup=TD_BARS_FOR_SETUP):
        result = pd.DataFrame(index=df.index)

        result['result'] = pd.Series.expanding(df['td_setup_start'], min_periods=1).sum()
        result['result'].replace(to_replace=np.NaN, value=0, inplace=True)

        return(result['result'].values.astype(int))

    #
    def build_td_sell_setup_id(df, bars_to_look_back=TD_SETUP_BARS_TO_LOOK_BACK, bars_to_setup=TD_BARS_FOR_SETUP):
        result = pd.DataFrame(index=df.index)

        result['buy_sell_setup_mask'] = df['td_setup'].where((df['td_setup_start'] == 1), 0)
        result['buy_sell_setup_mask'].replace(to_replace=0, method='ffill', inplace=True)

        result['result'] = df['td_setup_id'].where(result['buy_sell_setup_mask']==1, 0)

        return(result['result'].values.astype(int))

    def build_td_buy_setup_id(df, bars_to_look_back=TD_SETUP_BARS_TO_LOOK_BACK, bars_to_setup=TD_BARS_FOR_SETUP):
        result = pd.DataFrame(index=df.index)

        result['buy_sell_setup_mask'] = df['td_setup'].where((df['td_setup_start'] == 1), 0)
        result['buy_sell_setup_mask'].replace(to_replace=0, method='ffill', inplace=True)

        result['result'] = df['td_setup_id'].where(result['buy_sell_setup_mask']==-1, 0)

        return(result['result'].values.astype(int))

    def number_td_countdown_id(df):
        result = pd.DataFrame(index=df.index)
        result['abs_series'] = df.td_setup.abs()
        result['setup_end'] = (result.abs_series.where((result.abs_series == TD_BARS_FOR_SETUP), 0) / TD_BARS_FOR_SETUP)

        result['result'] = pd.Series.expanding(result.setup_end, min_periods=1).sum()
        result['result'].replace(to_replace=np.NaN, value=0, inplace=True)

        return(result['result'].values.astype(int))

    def find_odd_even_setups(df):
        result = pd.DataFrame(index=df.index)

        result['result'] = df['td_countdown_id'] % 2
        result['result'].replace(to_replace=np.NaN, value=0, inplace=True)
        return(result['result'].values.astype(int))

    # Start processing from pair 1 onwards
    def find_odd_even_pairs(df):
        result = pd.DataFrame(index=df.index)
        result['result'] = 2*(df['td_countdown_id']+1).floordiv(2, fill_value=0) - 1
        result['result'] = result['result'].where((result.result > 0), 0)
        return(result['result'].values.astype(int))

    # Start processing from pair 1 onwards
    def find_even_odd_pairs(df):
        result = pd.DataFrame(index=df.index)
        result['result'] = 2*(df['td_countdown_id']).floordiv(2, fill_value=0)
        return(result['result'].values.astype(int))

    def find_max_or_min_of_bars_six_and_seven(df):
        result = pd.DataFrame(index=df.index)
        # first find max or min of bars 6 or 7; i.e TD_BARS_FOR_SETUP-2 & TD_BARS_FOR_SETUP-3
        result['bear_n_minus_2'] = df['adj_low'].shift(2)
        result['bear_n_minus_3'] = df['adj_low'].shift(3)
        result['min'] = result[['bear_n_minus_2', 'bear_n_minus_3']].min(axis=1) # max along the rows
        # zero out the non-relevant entries
        result.loc[df[df.td_setup != -TD_BARS_FOR_SETUP].index, 'min'] = 0

        result['result'] = result['min']

        # Copy the results to bar8 and bar10-bar17
        #   copy the value of bar9 also to bars8
        result['result'].replace(to_replace=0, method='bfill', limit=1, inplace=True) # fill back
        #   copy the value of bar9 also to bars10-bars17
        result['result'].replace(to_replace=0, method='ffill', limit=(TD_BARS_FOR_SETUP-2), inplace=True) #fill forward

        return(result['result'].values)

    def find_setup_perfection(df):
        result = df.copy()

        # Find max or min of bars six and seven
        # first find max or min of bars 6 or 7; i.e TD_BARS_FOR_SETUP-2 & TD_BARS_FOR_SETUP-3
        result['bear_n_minus_2'] = df['adj_low'].shift(2)
        result['bear_n_minus_3'] = df['adj_low'].shift(3)
        result['min'] = result[['bear_n_minus_2', 'bear_n_minus_3']].min(axis=1) # max along the rows
        # zero out the non-relevant entries
        result.loc[df[df.td_setup != -TD_BARS_FOR_SETUP].index, 'min'] = 0


        result['max_or_min_bars_six_and_seven'] = result['min']

        # Copy the results to bar8 and bar10-bar17
        #   copy the value of bar9 also to bars8
        result['max_or_min_bars_six_and_seven'].replace(to_replace=0, method='bfill', limit=1, inplace=True) # fill back
        #   copy the value of bar9 also to bars10-bars17
        result['max_or_min_bars_six_and_seven'].replace(to_replace=0, method='ffill', limit=(TD_BARS_FOR_SETUP-2), inplace=True) #fill forward


        # bar 8 is defined as  "where next bar is bar9"
        buy_bar8 = result[result.td_setup.shift(-1) == -TD_BARS_FOR_SETUP]
        buy_bar9 = result[result.td_setup == -TD_BARS_FOR_SETUP]
        result['buy_bar8_perfection_shift1'] = 0
        result['buy_bar9_perfection'] = 0
        result.loc[buy_bar8.index, 'buy_bar8_perfection_shift1'] = (buy_bar8['adj_low'] <= buy_bar8['max_or_min_bars_six_and_seven']).shift(1).replace(to_replace=np.NaN, value=0)
        result.loc[buy_bar9.index, 'buy_bar9_perfection'] = (buy_bar9['adj_low'] <= buy_bar9['max_or_min_bars_six_and_seven'])
        result['buy_setup_perfection'] = (result['buy_bar8_perfection_shift1'] |
                                          result['buy_bar9_perfection']).replace(to_replace=np.NaN, value=0).astype(int)
        result['buy_previous_setup_perfection_found'] =  result['buy_setup_perfection'].replace(to_replace=np.NaN, value=0).astype(int)

        result['buy_previous_setup_perfection_found'].replace(to_replace=0, method='ffill', limit=(1), inplace=True)

        # If Setup Perfection is not found within the TD Setup, look 7 (TD_BARS_FOR_SETUP-2) bars beyond bar9
        # with each step looking at the previous one to see if it's found perfection
        for i in range(1, (TD_BARS_FOR_SETUP-1)):
            buy_bar_n = result[result.td_setup.shift(i) == -TD_BARS_FOR_SETUP]

            result.loc[buy_bar_n.index, 'buy_setup_perfection'] = ((buy_bar_n['adj_low'] <= buy_bar_n['max_or_min_bars_six_and_seven']) &
                                                                   (~buy_bar_n['buy_previous_setup_perfection_found']) ).astype(int)

            result['buy_previous_setup_perfection_found'] = (result['buy_setup_perfection'] |
                                                             result['buy_previous_setup_perfection_found']).astype(int)
            result['buy_previous_setup_perfection_found'].replace(to_replace=0, method='ffill', limit=(1), inplace=True)

        stock_df['buy_previous_setup_perfection_found'] = result['buy_previous_setup_perfection_found'] # Debug Code
        # Look for Setup Perfection if the previous bar doesn't have it.

        result['td_setup_perfection'] = result['buy_setup_perfection'] #
        result['td_setup_perfection'].replace(to_replace=np.NaN, value=0, inplace=True)

        return(result['td_setup_perfection'].values.astype(int))

    def find_tdst_sell(df, odd_even='odd'):
        result = df.copy()

        sells = df[df['td_setup'] > 0]

        # TDST Sell is the min of the
        g = sells.groupby('td_sell_setup_id')['adj_low']

        # Take the TDST from the previous sell_setup
        y = g.apply(lambda x: x.head(TD_BARS_FOR_SETUP).min())

        result['tdst_sell_'+odd_even] = result['td_countdown_id_'+odd_even].map(y)

        return(result['tdst_sell_'+odd_even])

    def find_tdst_buy(df, odd_even='odd'):
        result = df.copy()

        buys = df[df['td_setup'] < 0]

        # TDST Sell is the max of the
        g = buys.groupby('td_buy_setup_id')['adj_high']

        # Take the TDST from the previous sell_setup
        y = g.apply(lambda x: x.head(TD_BARS_FOR_SETUP).max())

        result['tdst_buy_'+odd_even] = result['td_countdown_id_'+odd_even].map(y)

        return(result['tdst_buy_'+odd_even])

    def find_buy_setup_flip(df):
        result = pd.DataFrame(index=df.index)
        buy_list = set(df.td_buy_setup_id.unique())
        buy_list.remove(0)
        buy_list = list(buy_list)

        sell_list = set(df.td_sell_setup_id.unique())
        sell_list.remove(0)
        sell_list = list(sell_list)

        result['sell_to_buy_flip'] = 0

        result['sell_to_buy_flip'] = (df.td_setup_id.isin(buy_list) & (df.td_setup_id-1).isin(sell_list)).astype(int)

        result['result'] = result['sell_to_buy_flip']

        return(result['result'].values.astype(int))

    def find_sell_setup_flip(df):
        result = pd.DataFrame(index=df.index)
        buy_list = set(df.td_buy_setup_id.unique())
        buy_list.remove(0)
        buy_list = list(buy_list)

        sell_list = set(df.td_sell_setup_id.unique())
        sell_list.remove(0)
        sell_list = list(sell_list)

        result['buy_to_sell_flip'] = 0

        result['buy_to_sell_flip'] = -(df.td_setup_id.isin(sell_list) & (df.td_setup_id-1).isin(buy_list)).astype(int)

        result['result'] = result['buy_to_sell_flip']

        return(result['result'].values.astype(int))

    def find_buy_stop_loss(df):
        result = pd.DataFrame(index = df.index)

        buys = df[df['td_buy_setup_id'] > 0] # pick only valid entries not 0
        result['td_buy_stop_loss'] = np.NaN # init

        if len(buys):
            # TDST Sell is the max of the
            g = buys.groupby('td_buy_setup_id')

            # TD Stop Loss is 'adj_low' - 'TrueRange'
            # TrueRange = 'adj_high' - 'adj_low'

            y = g.apply(lambda x: (2*x.head(TD_BARS_FOR_SETUP)['adj_low'].min()) -
                        x.loc[x.head(TD_BARS_FOR_SETUP)['adj_low'].idxmin(), 'adj_high'])
            result['td_buy_stop_loss'] = df['td_countdown_id'].map(y).replace(to_replace=np.NaN,
                                                                               method='ffill')
            if (y.shape[0] == 1):
                print('Debug find_buy_stop_loss:', df.bb_ticker.unique(), y)

        return(result['td_buy_stop_loss'].values)  

    def find_sell_stop_loss(df):
        result = pd.DataFrame(index = df.index)

        sells = df[df['td_sell_setup_id'] > 0] # pick only valid entries not 0
        result['td_sell_stop_loss'] = np.NaN # init

        if len(sells):
            # TDST Sell is the max of the
            g = sells.groupby('td_sell_setup_id')

            # TD Stop Loss is 'adj_high' + 'TrueRange'
            # TrueRange = 'adj_high' - 'adj_low'

            y = g.apply(lambda x: (2*x.head(TD_BARS_FOR_SETUP)['adj_high'].max()) -
                        x.loc[x.head(TD_BARS_FOR_SETUP)['adj_high'].idxmax(), 'adj_low'])
            if (y.shape[0] == 1):
                print('Debug find_sell_stop_loss:', df.bb_ticker.unique(), y)

            result['td_sell_stop_loss'] = df['td_countdown_id'].map(y).replace(to_replace=np.NaN,
                                                                               method='ffill')

        return(result['td_sell_stop_loss'].values)    

    def find_td_countdown(df, odd_even='odd'):
        result = df.copy()
        result['td_countdown_'+odd_even] = 0

        groups_countdown = result.groupby('td_countdown_id_'+odd_even, as_index=False) # start processing

        grouped_result = groups_countdown.apply(lambda x:
                                                (-1*(x['buy_N_close_lte_Nmin2_low_qualifier'].cumsum().clip_upper(TD_BARS_FOR_COUNTDOWN)) *
                                                                     x['buy_N_close_lte_Nmin2_low_qualifier']) ).reset_index(level=0, drop=True)
        if (grouped_result.shape[0] == 1): # if a single group
            grouped_result = grouped_result.transpose()

        # Convert multi-level index to single-level index
        result['td_countdown_'+odd_even] = grouped_result

        # replace 'bar13s' with bar130s until confirmed with completion criteria
        result['td_countdown_'+odd_even].replace(to_replace=TD_BARS_FOR_COUNTDOWN, value=TD_COUNTDOWN_INCOMPLETE_BAR_SYMBOL, inplace=True)
        result['td_countdown_'+odd_even].replace(to_replace=-TD_BARS_FOR_COUNTDOWN, value=-TD_COUNTDOWN_INCOMPLETE_BAR_SYMBOL, inplace=True)

        result['td_countdown_'+odd_even].replace(to_replace=np.NaN, value=0, inplace=True)

        stock_df['bar130_td_countdown_'+odd_even] = result['td_countdown_'+odd_even] # Debug

        #return(result['td_countdown_'+odd_even].values.astype(int))

        ### Look for TD_COUNTDOWN_COMPLETION

        '''
        Sell:
          (a) high13 >= close8,  &
          (b) close13 >= high two bars earlier bar13.shift(TD_COUNTDOWN_BARS_TO_LOOK_BACK) (high qualifier)

        Buy:
          (a) Low13 <= Close8, &
          (b) Close13 <= Low two bars earlier bar13.shift(TD_COUNTDOWN_BARS_TO_LOOK_BACK) (low qualifier)



          (c) More Conservative Approach:
             Buy: low8 <= close5
             Sell: high8 >= close5
        '''

        result['invalid_bars_mask'] = 0
        #result['sell_countdown_completion'] = 0
        result['buy_countdown_completion'] = 0
        #result['sell_countdown_bar8'] = 0
        #result['sell_countdown_bar5'] = 0

        grouped = result.groupby('td_countdown_id_'+odd_even, as_index=False)
        bar8_adj_close = grouped.apply(lambda x: (x['adj_close'].where(
            (x['td_countdown_'+odd_even].abs() == TD_COUNTDOWN_BAR8), 0)).replace(
                to_replace=0,method='ffill')).reset_index(level=0, drop=True)
        if (bar8_adj_close.shape[0] ==1):
            bar8_adj_close = bar8_adj_close.transpose()

        result['bar8_adj_close'] = bar8_adj_close

        '''# Sell
        sell_countdown_bar130 = result[result['td_countdown_'+odd_even] == (TD_COUNTDOWN_INCOMPLETE_BAR_SYMBOL)]

        # Check for perfection and mark bar13s
        if (len(sell_countdown_bar130) > 0):
            result.loc[sell_countdown_bar130.index, 'sell_countdown_completion'] = ((sell_countdown_bar130['adj_high'] >= sell_countdown_bar130['bar8_adj_close']) &
                                                                                    (sell_countdown_bar130['bar8_adj_close'] != 0)).astype(int)

        stock_df['sell_countdown_completion_'+odd_even] = result.sell_countdown_completion # Debug
    '''
        # Buy
        buy_countdown_bar130 = result[result['td_countdown_'+odd_even] == -TD_COUNTDOWN_INCOMPLETE_BAR_SYMBOL]

        # Check for perfection only if bar13s exist
        if (len(buy_countdown_bar130)>0):
            result.loc[buy_countdown_bar130.index, 'buy_countdown_completion'] = ((buy_countdown_bar130['adj_low'] <= buy_countdown_bar130['bar8_adj_close']) &
                                                                                  (buy_countdown_bar130['bar8_adj_close'] != 0)).astype(int)
        stock_df['buy_countdown_completion_'+odd_even] = result.buy_countdown_completion # Debug

        # Change all completed bar130s to bar13s
        '''result.loc[((result['sell_countdown_completion']==True) & (result['td_countdown_'+odd_even] == TD_COUNTDOWN_INCOMPLETE_BAR_SYMBOL)),
                   'td_countdown_'+odd_even] = TD_BARS_FOR_COUNTDOWN
        '''
        result.loc[((result['buy_countdown_completion']==True) & (result['td_countdown_'+odd_even] == -TD_COUNTDOWN_INCOMPLETE_BAR_SYMBOL)),
                   'td_countdown_'+odd_even] = -TD_BARS_FOR_COUNTDOWN


        stock_df['before_bar13_nixing_td_countdown_'+odd_even] = result['td_countdown_'+odd_even] # Debug

        # Remove everything after the first bar13

        bar13s_or_greater = result[result['td_countdown_'+odd_even].abs() == TD_BARS_FOR_COUNTDOWN][['td_countdown_id_'+odd_even, 'td_countdown_'+odd_even, ]]  
        group13s = bar13s_or_greater.groupby('td_countdown_id_'+odd_even, as_index=False)
        # first_bar13 = group13s['td_countdown_'+odd_even].apply(lambda x: x.tail(-1))
        first_bar13 = group13s['td_countdown_'+odd_even].apply(lambda x: x.head(1)).reset_index(level=0, drop=True)
        if (first_bar13.shape[0] == 1):
            first_bar13 = first_bar13.transpose()
        result['first_bar13_'+odd_even] = 0 # init
        result['bar13s_to_remove_'+odd_even] = 0
        if (len(first_bar13)):
            result.loc[first_bar13.index, 'first_bar13_'+odd_even] = 1
            group_first_bar13 = result.groupby('td_countdown_id_'+odd_even)['first_bar13_'+odd_even]
            result['first_bar13_fill_'+odd_even] = group_first_bar13.apply(lambda x: x.replace(to_replace=0, method='ffill'))
            result['bar13s_to_remove_'+odd_even] = result['first_bar13_fill_'+odd_even] ^ result['first_bar13_'+odd_even]
            stock_df['bar13s_to_remove_'+odd_even] = result['bar13s_to_remove_'+odd_even] # Debug
        result['td_countdown_'+odd_even] = result['td_countdown_'+odd_even].where((result['bar13s_to_remove_'+odd_even] == 0), 0) #remove the bar13s and bar130s after the first bar13

        '''
         # Cancellation of Countdown
         Cancel an incomplete TD Buy Countdown:
         1. if the price action rallies and generates a TD Sell Setup, or
         2. if the market trades higher and posts a true low above the true high of the prior
            TD Buy Setupâ€”that is, TDST-buy resistance td_prev_tdst_buy

         Cancel an incomplete TD Sell Countdown
         1. Price actions leads to   sell-off and generates a TD Buy Setup, or
         2. if the market trades lower and posts a true high below the true low of the prior TD-sell-setup td_prev_tdst_sell
        '''

        '''# Cancel condition 1
        #set_sell_setup_cancels_countdown_ids = set(result[result['td_sell_setup_flip'] != 0]['td_setup_id']-1)
        set_buy_setup_cancels_countdown_ids  = set(result[result['td_buy_setup_flip'] != 0]['td_setup_id']-1)
        set_setup_cancels_countdown_ids = set_buy_setup_cancels_countdown_ids - {0}

            # Look for a bar 13 in the countdown portion overlapping the setup_id
        countdown_entries_to_check = result[result['td_setup_id'].isin(list(set_sell_setup_cancels_countdown_ids)) &
                                            result['td_countdown_id_'+odd_even].isin(list(set_sell_setup_cancels_countdown_ids))]
        set_valid_countdown_ids = set(countdown_entries_to_check[countdown_entries_to_check['td_countdown_'+odd_even].abs() == TD_BARS_FOR_COUNTDOWN]['td_countdown_id_'+odd_even] )
            # Remove the valid countdowns from the to-be-cancelled countdowns
        set_setup_cancels_countdown_ids = set_setup_cancels_countdown_ids - set_valid_countdown_ids

        #print('Debug: Valid Countdown Ids', len(set_valid_countdown_ids), set_valid_countdown_ids) # Debug
        #print('Debug: Length set_setup_cancels_countdown_ids', len(set_setup_cancels_countdown_ids), set_setup_cancels_countdown_ids) # Debug

        stock_df['setup_cancelled_countdown_'+odd_even] = 0 #Debug
        setup_cancelled_countdowns = result[result['td_countdown_id_'+odd_even].isin(list(set_setup_cancels_countdown_ids))]
        stock_df.loc[setup_cancelled_countdowns.index, 'setup_cancelled_countdown_'+odd_even] = 1 #Debug'''

        # Cancel condition 2
        result['continuous_td_countdown_'+odd_even] = result['td_countdown_'+odd_even].replace(to_replace=0, method='ffill')
        buy_countdowns = result[(result['continuous_td_countdown_'+odd_even] < 0) &
                                (result['continuous_td_countdown_'+odd_even] > -TD_BARS_FOR_COUNTDOWN)] # Find the buy countdowns below
        set_higher_market_cancels_countdown_ids = set(buy_countdowns[buy_countdowns['adj_low'] >
                                                               buy_countdowns['td_tdst_buy_setup_high_'+odd_even]]['td_countdown_id_'+odd_even].unique())

        stock_df['higher_market_cancels_countdown_'+odd_even] = 0 # Debug
        stock_df.loc[buy_countdowns[buy_countdowns['adj_low'] >
                                    buy_countdowns['td_tdst_buy_setup_high_'+odd_even]].index,
                     'higher_market_cancels_countdown_'+odd_even] = 1 # Debug

        '''sell_countdowns = result[(result['continuous_td_countdown_'+odd_even] > 0) &
                                 (result['continuous_td_countdown_'+odd_even] < TD_BARS_FOR_COUNTDOWN) ] # Find the sell countdowns
        set_lower_market_cancels_countdown_ids = set(sell_countdowns[sell_countdowns['adj_high'] <
                                                                sell_countdowns['td_tdst_sell_setup_low_'+odd_even]]['td_countdown_id_'+odd_even].unique())

        stock_df['lower_market_cancels_countdown_'+odd_even] = 0 # Debug
        stock_df.loc[sell_countdowns[sell_countdowns['adj_high'] <
                                              sell_countdowns['td_tdst_sell_setup_low_'+odd_even]].index,
                              'lower_market_cancels_countdown_'+odd_even] = 1 # Debug'''

        list_cancelled_countdown_ids = list(set_higher_market_cancels_countdown_ids)  # option removed: 'set_setup_cancels_countdown_ids |'

        cancelled_countdown_rows = result[result['td_countdown_id_'+odd_even].isin(list_cancelled_countdown_ids) ]

        stock_df['original_td_countdown_'+odd_even] = result['td_countdown_'+odd_even] #Debug

        result.loc[cancelled_countdown_rows.index, 'td_countdown_'+odd_even] = 0

        return(result[['td_countdown_'+odd_even]].values.astype(int))

    def merge_td_countdown(df):
        result = pd.DataFrame(index=df.index)

        result['td_countdown'] = df['td_countdown_even'].where( (df['td_countdown_even'].abs() > df['td_countdown_odd'].abs()),
                                                                df['td_countdown_odd'])

        return(result['td_countdown'].values.astype(int))

    def find_conservative_countdown_completion(df, odd_even='odd'):
        result = df.copy()

        '''
        #for conservative countdown completion
        result['sell_countdown_adj_high_bar8'] = (result['adj_high'].where((result['td_countdown'] == TD_COUNTDOWN_BAR8), 0)).replace(to_replace=0, method='ffill')
        result['sell_countdown_adj_close_bar5'] = (result['adj_close'].where((result['td_countdown'] == TD_COUNTDOWN_BAR5), 0)).replace(to_replace=0, method='ffill')

        result.loc[sell_countdown_bar13.index, 'sell_conservative_countdown_completion'] = (result['sell_countdown_completion'] &
                                                            (sell_countdown_bar13['sell_countdown_adj_high_bar8'] >= sell_countdown_bar13['sell_countdown_adj_close_bar5']) ).astype(int)

        result['buy_conservative_countdown_completion'] = (result['buy_countdown_completion'] &
                                                            (buy_countdown_bar8['adj_low'] <= buy_countdown_bar5['adj_close']) ).astype(int)
        '''
        return(result['td_countdown_'+odd_even]) # just a place holder


    #print(stock_df[['adj_close', 'price_flip', 'td_setup']])
    #Forward Returns Computed
    stock_df['abs_return_1day'] = stock_df.adj_close.pct_change(periods=1).shift((-1))
    stock_df['abs_return_2day'] = stock_df.adj_close.pct_change(periods=2).shift((-2))
    stock_df['abs_return_1wk'] = stock_df.adj_close.pct_change(periods=WEEK_PERIOD).shift((-WEEK_PERIOD))
    stock_df['abs_return_2wk'] = stock_df.adj_close.pct_change(periods=(2*WEEK_PERIOD)).shift(-2*(WEEK_PERIOD))
    stock_df['abs_return_1mo'] = stock_df.adj_close.pct_change(periods=(MONTH_PERIOD)).shift((-MONTH_PERIOD))

    
    stock_df['price_flip'] = 0
    stock_df['td_setup'] = 0

    stock_df['price_flip'] = find_price_flip(stock_df) #buy only
    stock_df['td_setup'] = find_td_setup(stock_df, bars_to_look_back=TD_SETUP_BARS_TO_LOOK_BACK, bars_to_setup=TD_BARS_FOR_SETUP)
    stock_df['td_setup_start'] = find_setup_start(stock_df) #ok
    stock_df['td_setup_id'] = number_td_setups(stock_df) #ok
    #stock_df['td_sell_setup_id'] = build_td_sell_setup_id(stock_df) 
    stock_df['td_buy_setup_id'] = build_td_buy_setup_id(stock_df) #ok

    # Find a  setup flip - change from buy->sell or sell->buy
    #stock_df['td_sell_setup_flip'] = find_sell_setup_flip(stock_df)
    #stock_df['td_buy_setup_flip'] = find_buy_setup_flip(stock_df) #ok

    # Find Setup Perfection
    #stock_df['max_or_min_bars_six_and_seven'] = find_max_or_min_of_bars_six_and_seven(stock_df)
    stock_df['td_setup_perfection'] = find_setup_perfection(stock_df) #ok

    # Find TD Countdown
    stock_df['td_countdown_id'] = number_td_countdown_id(stock_df) #ok
    stock_df['td_countdown_id_odd'] = find_odd_even_pairs(stock_df) #ok
    stock_df['td_countdown_id_even'] = find_even_odd_pairs(stock_df) #ok

    # Find the Sell and Buy Stop Loss
    #stock_df['td_sell_stop_loss'] = find_sell_stop_loss(stock_df) 
    stock_df['td_buy_stop_loss'] = find_buy_stop_loss(stock_df) #ok

    # Find TDST Resistance Lines
    #stock_df['td_tdst_sell_setup_low_odd'] = find_tdst_sell(stock_df, odd_even='odd')
    #stock_df['td_tdst_sell_setup_low_even'] = find_tdst_sell(stock_df, odd_even='even')
    stock_df['td_tdst_buy_setup_high_odd'] = find_tdst_buy(stock_df, odd_even='odd') #ok
    stock_df['td_tdst_buy_setup_high_even'] = find_tdst_buy(stock_df, odd_even='even') #ok

    '''stock_df['sell_N_close_gte_Nmin2_high_qualifier'] = (stock_df.adj_close >=
                                                         stock_df.adj_high.shift(TD_COUNTDOWN_BARS_TO_LOOK_BACK)).replace(to_replace=np.NaN,
                                                                                                                    value=0).astype(int)
    '''
    stock_df['buy_N_close_lte_Nmin2_low_qualifier'] = (stock_df.adj_close <=
                                                     stock_df.adj_low.shift(TD_COUNTDOWN_BARS_TO_LOOK_BACK)).replace(to_replace=np.NaN,
                                                                                                                    value=0).astype(int)
    stock_df['td_countdown_odd'] = find_td_countdown(stock_df, odd_even='odd')
    stock_df['td_countdown_even'] = find_td_countdown(stock_df, odd_even='even')

    stock_df['td_countdown'] = merge_td_countdown(stock_df)
    
    ###################################################### RSI + MACD #################################################
    
    stock_groups = stock_df.groupby('bb_ticker', as_index=False)['adj_close']
    stock_df['rsi_14day'] = stock_groups.apply(lambda x: pd.DataFrame(talib.RSI(x.values, 14), index=x.index))
    stock_df['rsi_21day'] = stock_groups.apply(lambda x: pd.DataFrame(talib.RSI(x.values, 21), index=x.index))
    stock_df['rsi_50day'] = stock_groups.apply(lambda x: pd.DataFrame(talib.RSI(x.values, 50), index=x.index))
    stock_df['macd'] = 0
    stock_df['macd_signal'] = 0
    stock_df['macd_hist'] = 0
    macd, macd_signal, macd_hist = talib.MACD(stock_df.adj_close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    #macd.index = macd.index.get_level_values(level=1)
    stock_df['macd'] = macd.astype(float)
    #macd_signal.index = macd_signal.index.get_level_values(level=1)
    stock_df['macd_signal'] = macd_signal.astype(float)
    #macd_hist.index = macd_hist.index.get_level_values(level=1)
    stock_df['macd_hist'] = macd_hist.astype(float)

    #stock_df.to_excel('RSI_MACD_28.xlsx')
    stock_groups = stock_df.groupby('bb_ticker')

    print('Computing Crossovers')
    stock_df['macd_buy_crossover'] = stock_groups.macd_hist.apply(lambda x: ((x >= 0) & (x.shift(1) < 0)).astype(int))
    stock_df['macd_sell_crossover'] = stock_groups.macd_hist.apply(lambda x: ((x < 0) & (x.shift(1) >= 0)).astype(int))
    stock_df['rsi_buy_crossover'] = stock_groups.rsi_14day.apply(lambda x: ((x <= RSI_BUY_LEVEL) & (x.shift(1) > RSI_BUY_LEVEL)).astype(int))
    stock_df['rsi_sell_crossover'] = stock_groups.rsi_14day.apply(lambda x: ((x >= RSI_SELL_LEVEL) & (x.shift(1) < RSI_SELL_LEVEL)).astype(int))
    stock_df['rsi_buy_crossover_50'] = stock_groups.rsi_14day.apply(lambda x: ((x <= 50) & (x.shift(1) > 50)).astype(int))
    stock_df['rsi_sell_crossover_50'] = stock_groups.rsi_14day.apply(lambda x: ((x >= 50) & (x.shift(1) < 50)).astype(int))

    stock_df['macd_buy_rolling'] = stock_groups.macd_buy_crossover.apply(lambda x: x.rolling(window=MACD_RSI_WINDOW).sum())
    stock_df['macd_sell_rolling'] = stock_groups.macd_sell_crossover.apply(lambda x: x.rolling(window=MACD_RSI_WINDOW).sum())
    stock_df['rsi_buy_rolling'] = stock_groups.rsi_buy_crossover.apply(lambda x: x.rolling(window=MACD_RSI_WINDOW).sum())
    stock_df['rsi_sell_rolling'] = stock_groups.rsi_sell_crossover.apply(lambda x: x.rolling(window=MACD_RSI_WINDOW).sum())

    return(stock_df)

#OK



# In[3]:

#GET_STATS

'''
bootstrap: test the ci of the population (with replacement)
'''
def bootstrap(data, num_samples, statistic, alpha): # use : bootstrap(data['F_ols_sp400_alpha_2q].values, 1000, np.mean, 0.05)
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    if (n > 0):
        idx = np.random.randint(0, n, (num_samples, n))
        samples = data[idx]
        stat = np.sort(statistic(samples, 1))
        return (stat[int((alpha/2.0)*num_samples)],
                stat[int((1-alpha/2.0)*num_samples)])
    else:
        return(np.NaN, np.NaN)

'''
monte_carlo: test the bounds of a selected population (selection without replacement)
'''
def monte_carlo(data, n,num_samples, statistic, alpha):
    """Returns monte carlo simulation estimate of 100.0*(1-alpha) CI for statistic."""
    if (len(data)):
        stat = np.sort(np.array([np.mean(np.random.choice(data,size = n,replace=False)) for i in range (0,num_samples)]))
        return (stat[int((alpha/2.0)*num_samples)],
                stat[int((1-alpha/2.0)*num_samples)])
    else:
        return(np.NaN, np.NaN)

def get_stats(data, pred, strategy, buy_or_sell, result):

    if (len(pred)):
        # This is the base case
        print(strategy, buy_or_sell, 'Predictions=', '\n    Number of Predictions', len(pred))
        sys.stdout.flush()

        for p in PERIODS:

            valid_entries = data[pd.notnull(data['abs_return_'+p])]
            valid_pred = pred[pd.notnull(pred['abs_return_'+p])]
            hit = valid_pred[valid_pred['abs_return_'+p] >= 0] if (buy_or_sell == 'buy') else (
                valid_pred[valid_pred['abs_return_'+p] < 0] )
            hit_rate = len(hit)/len(valid_pred) #precision

            random_hit = valid_entries[valid_entries['abs_return_'+p] >=0] if (buy_or_sell == 'buy') else (
                valid_entries[valid_entries['abs_return_'+p] < 0] )
            ingoing_hit_rate = len(random_hit)/len(valid_entries)

            result.loc[strategy+' '+buy_or_sell+' '+p, '#Pred'] = len(valid_pred)
            result.loc[strategy+' '+buy_or_sell+' '+p, 'HR'] = hit_rate
            result.loc[strategy+' '+buy_or_sell+' '+p, 'In_HR'] = ingoing_hit_rate
            result.loc[strategy+' '+buy_or_sell+' '+p, 'Lift'] = hit_rate - ingoing_hit_rate

            monte_carlo_result = monte_carlo(valid_entries['abs_return_'+p].values, len(valid_pred),
                                             NUMBER_OF_RANDOM_SAMPLES, np.mean, (1-CONFIDENCE_INTERVAL))  #Monte Carlo 1 week

            result.loc[strategy+' '+buy_or_sell+' '+p, 'in_low'] = ilow = monte_carlo_result[0] if (buy_or_sell == 'buy')                                                                                                 else -monte_carlo_result[0]
            result.loc[strategy+' '+buy_or_sell+' '+p, 'in_mean'] = imean = valid_entries['abs_return_'+p].mean() if                                                    (buy_or_sell == 'buy') else ( -valid_entries['abs_return_'+p].mean())
            result.loc[strategy+' '+buy_or_sell+' '+p, 'in_high'] = ihigh = monte_carlo_result[1] if (buy_or_sell == 'buy')                                                                                                else -monte_carlo_result[1]

            # This is the algorithm
            bootstrap_result = bootstrap(valid_pred['abs_return_'+p].values,
                                         NUMBER_OF_RANDOM_SAMPLES, np.mean, (1-CONFIDENCE_INTERVAL))

            result.loc[strategy+' '+buy_or_sell+' '+p, 'out_low'] = olow = bootstrap_result[0] if (buy_or_sell == 'buy') else -bootstrap_result[0]
            result.loc[strategy+' '+buy_or_sell+' '+p, 'out_mean'] = omean = valid_pred['abs_return_'+p].mean() if (buy_or_sell == 'buy') else -valid_pred['abs_return_'+p].mean()
            result.loc[strategy+' '+buy_or_sell+' '+p, 'out_high'] = ohigh = bootstrap_result[1] if (buy_or_sell == 'buy') else -bootstrap_result[1]
            result.loc[strategy+' '+buy_or_sell+' '+p, 'ExcessRet'] = omean - imean

            # Write intermediate results
            print(result.to_string())
            sys.stdout.flush()

    else:
        print('Error in get_stats: No Predictions')
        sys.stdout.flush()
        
    # Write back the results
    result.to_csv('PT_result_demark.csv')
    return(result)


# In[4]:

if __name__ == "__main__":

    print('Starting DeMark, RSI and MACD Simulation:')
    data_df = pd.read_excel('DataTwentyEight.xlsx') # Test dataframe
    print('Number of stocks in database = ', len(data_df.bb_ticker.unique()))
   
    print('Length Input Data:', len(data_df))
    sys.stdout.flush()
    result_output = pd.DataFrame()

    print('Confidence Interval:', CONFIDENCE_INTERVAL)

    # Compute DeMark
    list_of_securities = list(data_df.bb_ticker.unique())
    
    sys.stdout.flush()
 
    gdf = data_df.groupby('bb_ticker')
    demark_df = gdf.apply(lambda x: demark(x))
            
    # Write DeMark to file
    demark_df.to_excel('AllSignals.xlsx')
    print('Written AllSignals.xlsx with the DeMark, RSI and MACD signals.')
    sys.stdout.flush()


# In[5]:

# Throw out the NaN's
valid_entry = demark_df[pd.notnull(demark_df.td_countdown)]
valid_entries_buy = demark_df[(pd.notnull(demark_df.macd_buy_rolling) &                               pd.notnull(demark_df.rsi_buy_rolling) &                               pd.notnull(demark_df.rsi_buy_crossover) &                               pd.notnull(demark_df.macd_buy_crossover) &                              pd.notnull(demark_df.rsi_50day))]

# STRATEGIES
### RSI + MACD
demark_df.loc[valid_entries_buy.index, 'macd_rsi_buy'] = ((valid_entries_buy['macd_buy_crossover'] & valid_entries_buy['rsi_buy_rolling']) |
                                                          (valid_entries_buy['rsi_buy_crossover'] & valid_entries_buy['macd_buy_rolling'])).astype(int)

macd_rsi_buy_pred = demark_df[demark_df.macd_rsi_buy == 1]
get_stats(valid_entries_buy, macd_rsi_buy_pred, 'RSI_MACD', 'buy', result_output)

### demark
demark_buy = demark_df[demark_df.td_countdown == -TD_BARS_FOR_COUNTDOWN]
get_stats(valid_entry, demark_buy, 'demark', 'buy', result_output)

### TD Setup 1 - Accumulation and Distribution - Check if there's momentum in the Setup
demark_buy = demark_df[demark_df.td_setup == -1]
get_stats(valid_entry, demark_buy, 'demark-setup1', 'buy', result_output)

### TD Setup 2 - Accumulation and Distribution - Check if there's momentum in the Setup
demark_buy = demark_df[demark_df.td_setup == -2]
get_stats(valid_entry, demark_buy, 'demark-setup2', 'buy', result_output)

 ### TD Setup 3 - Accumulation and Distribution - Check if there's momentum in the Setup
demark_buy = demark_df[demark_df.td_setup == -3]
get_stats(valid_entry, demark_buy, 'demark-setup3', 'buy', result_output)

### TD Setup 4 - Accumulation and Distribution - Check if there's momentum in the Setup
demark_buy = demark_df[demark_df.td_setup == -4]
get_stats(valid_entry, demark_buy, 'demark-setup4', 'buy', result_output)

### TD Setup 6 - Accumulation and Distribution - Check if there's momentum in the Setup
demark_buy = demark_df[demark_df.td_setup == -6]
get_stats(valid_entry, demark_buy, 'demark-setup6', 'buy', result_output)

### TD Setup 9 - Accumulation and Distribution - Check if there's momentum in the Setup
demark_buy = demark_df[demark_df.td_setup == -TD_BARS_FOR_SETUP]
get_stats(valid_entry, demark_buy, 'demark-setup9', 'buy', result_output)

### TD Countdown 2 - Accumulation and Distribution - Check if there's momentum in the Countdown
demark_buy = demark_df[demark_df.td_countdown == -2]
get_stats(valid_entry, demark_buy, 'demark-countdown2', 'buy', result_output)

### TD Countdown 3 - Accumulation and Distribution - Check if there's momentum in the Countdown
demark_buy = demark_df[demark_df.td_countdown == -3]
get_stats(valid_entry, demark_buy, 'demark-countdown3', 'buy', result_output)

 ### TD Countdown 5 - Accumulation and Distribution - Check if there's momentum in the Countdown
demark_buy = demark_df[demark_df.td_countdown == -5]
get_stats(valid_entry, demark_buy, 'demark-countdown5', 'buy', result_output)

### TD Countdown 8 - Accumulation and Distribution - Check if there's momentum in the Countdown
demark_buy = demark_df[demark_df.td_countdown == -8]
get_stats(valid_entry, demark_buy, 'demark-countdown8', 'buy', result_output)

### TD Countdown 130 - Accumulation and Distribution - Check if there's momentum in the Countdown
demark_buy = demark_df[demark_df.td_countdown == -130]
get_stats(valid_entry, demark_buy, 'demark-countdown130', 'buy', result_output)

### TD Countdown 13 - Accumulation and Distribution - Check if there's momentum in the Countdown
demark_buy = demark_df[demark_df.td_countdown == -TD_BARS_FOR_COUNTDOWN]
get_stats(valid_entry, demark_buy, 'demark-countdown13', 'buy', result_output)


# # End. Enjoy.
