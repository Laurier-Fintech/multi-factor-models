# multi-factor-models

Risk model order
1. Factor generation
2. Factor baskets
3. Matrix Construction
4. Stock Returns


Alpha model order
1. Build_signals (run this anytime there is a change to crsp_clean or new signals are added), calls everything in the signal_lib folder
2. run_alpha_model lets you test the alpha model over a fixed time window, it will automatically divide up the period into 80% test, 10% validation, and 10% testing period


In the backtesting folder, the pipeline goes as follows: 
1. Portfolio_weights.py will first first run the alpha model given a start and end date and get a vector of each cusips expected returns and alpha score
2. Alpha scores are then passed in to the risk model in which systematic and stock specific risk is calculated and a vector of weights is produced for that start and end date set. 

3. In backtesting_framework.py, the process above is conducted every X days and returns are calculated over the entire window period. This follows an unanchored-walk forward methodology as to similate live trading without look-ahead bias