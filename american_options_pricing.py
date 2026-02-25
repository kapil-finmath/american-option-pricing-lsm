import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------
# 1. Download Market Data
# ---------------------------
ticker = "AAPL"

data = yf.download(ticker, start="2020-01-01", progress=False)

# Flatten columns if MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Choose best price column safely
if "Adj Close" in data.columns:
    price_series = data["Adj Close"]
elif "Close" in data.columns:
    price_series = data["Close"]
else:
    raise ValueError("No usable price column found")

S0 = price_series.iloc[-1]

# Compute volatility from returns
returns = price_series.pct_change().dropna()
sigma = returns.std() * np.sqrt(252)

print("Spot =", S0)
print("Volatility =", sigma)

# ---------------------------
# 2. Model Parameters
# ---------------------------
r = 0.05
T = 1
steps = 50
paths = 1000
dt = T / steps

# ---------------------------
# 3. Simulate GBM Paths
# ---------------------------
S = np.zeros((paths, steps + 1))
S[:, 0] = S0

for t in range(1, steps + 1):
    Z = np.random.normal(size=paths)
    S[:, t] = S[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

# ---------------------------
# 4. Plot Some Paths
# ---------------------------
plt.figure(figsize=(10,6))
for i in range(20):
    plt.plot(S[i])
plt.title("Simulated Stock Price Paths")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.show()

# ---------------------------
# 5. LSM American Put Pricing
# ---------------------------
K = S0
payoff = np.maximum(K - S[:, -1], 0)

for t in range(steps-1, 0, -1):

    St = S[:, t]
    itm = np.where(K - St > 0)[0]

    if len(itm) == 0:
        continue

    X = St[itm]
    Y = payoff[itm] * np.exp(-r*dt)

    # Regression basis: 1, S, S^2
    A = np.vstack([np.ones(len(X)), X, X**2]).T
    coeff = np.linalg.lstsq(A, Y, rcond=None)[0]

    continuation = coeff[0] + coeff[1]*St + coeff[2]*St**2
    exercise = np.maximum(K - St, 0)

    exercise_now = exercise > continuation

    payoff[exercise_now] = exercise[exercise_now]
    payoff[~exercise_now] *= np.exp(-r*dt)

price = np.mean(payoff) * np.exp(-r*dt)

print("American Put Price =", price)