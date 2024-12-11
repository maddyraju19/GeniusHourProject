import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings


warnings.filterwarnings("ignore")
params = []
ratings = [pd.read_csv("/workspaces/GeniusHourProject/Drew Brees QB Data.csv"),
    pd.read_csv("/workspaces/GeniusHourProject/Joe Montana QB Data.csv"),
    pd.read_csv("/workspaces/GeniusHourProject/John Elway QB Data.csv"),
    pd.read_csv("/workspaces/GeniusHourProject/Tom Brady QB Data.csv"), 
    pd.read_csv("/workspaces/GeniusHourProject/Patrick Mahomes QB Data.csv")]
volatility = 0 
for i in range(len(ratings)):
    qbrate = list(ratings[i]["Rate"])
    for j in range(len(qbrate)):
        try:
            qbrate[j] = float(qbrate[j])
        except ValueError:
            qbrate[j] = None
    qbrate = qbrate[1:]
    qbrate = pd.Series(qbrate).interpolate()  
    diff = qbrate.diff().diff().dropna()
    if i < 4:
        model = ARIMA(diff, order=(2, 0, 0))
        model_fit = model.fit()
        params.append([float(model_fit.params["const"]), float(model_fit.params["ar.L1"]), float(model_fit.params["ar.L2"]), float(model_fit.params["sigma2"])])
        volatility += np.std(diff) / 4
    else:
        pm_rating = diff


avg_params = [0, 0, 0, 0]
for i in range(4):
    for j in range(4):
        avg_params[j] += params[i][j] / 4

pm_model = ARIMA(pm_rating, order=(2, 0, 0))
pm_fit = pm_model.fit(start_params=pd.Series(avg_params))

num_simulations = 1000
num_future_years = 20  

prime_years = np.zeros(num_simulations)


for i in range(num_simulations):
    forecasted_values = pm_fit.forecast(steps=num_future_years)
    random_errors = np.random.normal(scale=volatility, size=len(forecasted_values))
    simulated_values = forecasted_values + random_errors
    for j in range(2, num_future_years):
        if np.mean(simulated_values[j-3:j]) <= -10:
            prime_years[i] = j
            break

avg_years = round(np.mean(prime_years))

print(f"Patrick Mahomes will peak in {str(avg_years + 2024)} at the age of {str(avg_years + 29)}")
