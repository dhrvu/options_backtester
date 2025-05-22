#Option#
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import math
from scipy.stats import norm
import plotly.express as px

# ---------------------------------------------------
# 1. Sidebar: User Inputs
# ---------------------------------------------------
st.sidebar.title("Options Strategy Backtester")

ticker_symbol = st.sidebar.text_input(
    "Underlying Ticker (e.g., AAPL, TSLA):", 
    value="AAPL"
).upper()

strategy_type = st.sidebar.selectbox(
    "Strategy Type:",
    ("Straddle", "Strangle", "Iron Condor")
)

entry_mode = st.sidebar.radio(
    "Entry Rule:",
    ("Fixed Date Range", "15th of Every Month")
)

if entry_mode == "Fixed Date Range":
    start_date = st.sidebar.date_input(
        "Start Date:", 
        value=datetime.date(2020, 1, 1)
    )
    end_date = st.sidebar.date_input(
        "End Date:", 
        value=datetime.date(2023, 12, 31)
    )
else:
    start_date = None
    end_date = None

days_to_expiry = st.sidebar.number_input(
    "Days to Expiry (e.g., 30):", 
    min_value=1, max_value=180, value=30, step=1
)

risk_free_rate = st.sidebar.number_input(
    "Risk-free Rate (annual, decimal):", 
    min_value=0.0, max_value=0.10, value=0.015, format="%.4f"
)

vol_choice = st.sidebar.selectbox(
    "Volatility Estimate:",
    ("Realized (21-day)", "Fixed (enter manually)")
)
if vol_choice == "Fixed (enter manually)":
    fixed_vol = st.sidebar.number_input(
        "Fixed Annualized Volatility (decimal):", 
        min_value=0.01, max_value=2.0, value=0.25, format="%.4f"
    )
else:
    fixed_vol = None

if strategy_type == "Strangle":
    strangle_pct = st.sidebar.slider(
        "Percent Offset for Strikes (e.g., 5%):", 
        min_value=1, max_value=20, value=5
    )
elif strategy_type == "Iron Condor":
    ic_lower_put_pct = st.sidebar.slider(
        "Lower Put Strike Offset (%) below ATM:", 
        min_value=1, max_value=20, value=10
    )
    ic_outer_put_pct = st.sidebar.slider(
        "Inner Put Strike Offset (%) below ATM:", 
        min_value=1, max_value=20, value=5
    )
    ic_inner_call_pct = st.sidebar.slider(
        "Inner Call Strike Offset (%) above ATM:", 
        min_value=1, max_value=20, value=5
    )
    ic_outer_call_pct = st.sidebar.slider(
        "Upper Call Strike Offset (%) above ATM:", 
        min_value=1, max_value=20, value=10
    )

run_button = st.sidebar.button("Run Backtest")

# ---------------------------------------------------
# 2. Main Area: Title & Description
# ---------------------------------------------------
st.title("Options Strategy Backtester")
st.write("""
This app backtests your chosen options strategy over historical data.\nWe simulate theoretical option prices at entry using the Black-Scholes formula\n(either based on 21-day realized volatility or a fixed volatility you enter),\nthen compute the actual payoff at expiry (intrinsic value based on underlying price).\nFinally, we display per-trade P/L, an equity curve, and summary metrics.
""")

if not run_button:
    st.write("Adjust parameters on the left and click **Run Backtest** to see results.")
    st.stop()

# ---------------------------------------------------
# 3. Data Download & Preparation
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def load_price_data(symbol: str, start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    if end is None:
        hist = ticker.history(start=start)
    else:
        hist = ticker.history(start=start, end=end)
    hist.reset_index(inplace=True)
    # Convert to datetime and remove timezone (if any)
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist.set_index("Date", inplace=True)
    return hist

if entry_mode == "Fixed Date Range":
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")
else:
    start_str = "2015-01-01"
    end_str   = datetime.date.today().strftime("%Y-%m-%d")

df_price = load_price_data(ticker_symbol, start=start_str, end=end_str)

if vol_choice == "Realized (21-day)":
    df_price["log_return"] = np.log(df_price["Close"] / df_price["Close"].shift(1))
    df_price["RealizedVol"] = df_price["log_return"].rolling(window=21).std() * np.sqrt(252)

# ---------------------------------------------------
# 4. Utility Functions
# ---------------------------------------------------
def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    if sigma <= 0 or T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def round_strike_to_dollar(price: float) -> int:
    return max(1, int(round(price)))

def get_nth_future_trading_date(df: pd.DataFrame, entry_date: pd.Timestamp, days_forward: int) -> pd.Timestamp:
    all_dates = df.index.tolist()
    if entry_date not in all_dates:
        future_dates = [d for d in all_dates if d > entry_date]
        if not future_dates:
            return None
        entry_date = future_dates[0]
    idx = all_dates.index(entry_date)
    target_idx = idx + days_forward
    if target_idx >= len(all_dates):
        return None
    return all_dates[target_idx]

# ---------------------------------------------------
# 5. Backtesting Loop
# ---------------------------------------------------
results = []

if entry_mode == "Fixed Date Range":
    all_trading_dates = [d for d in df_price.index if (d.date() >= start_date and d.date() <= end_date)]
    entry_dates = []
    if all_trading_dates:
        cursor_date = all_trading_dates[0]
        while True:
            entry_dates.append(cursor_date)
            pos = all_trading_dates.index(cursor_date)
            next_pos = pos + days_to_expiry
            if next_pos >= len(all_trading_dates):
                break
            cursor_date = all_trading_dates[next_pos]
else:
    entry_dates = []
    min_year = df_price.index[0].year
    max_year = df_price.index[-1].year
    for y in range(min_year, max_year + 1):
        for m in range(1, 13):
            candidate = pd.Timestamp(f"{y}-{m:02d}-15")
            if candidate not in df_price.index:
                future = [d for d in df_price.index if d >= candidate]
                if not future:
                    continue
                candidate = future[0]
            if candidate in df_price.index:
                entry_dates.append(candidate)

for entry_date in entry_dates:
    expiry_date = get_nth_future_trading_date(df_price, entry_date, days_to_expiry)
    if expiry_date is None:
        continue
    S0 = df_price.loc[entry_date, "Close"]
    ST = df_price.loc[expiry_date, "Close"]
    if vol_choice == "Realized (21-day)":
        sigma = df_price.loc[entry_date, "RealizedVol"]
        if pd.isna(sigma):
            continue
    else:
        sigma = fixed_vol
    T = days_to_expiry / 252
    if strategy_type == "Straddle":
        K = round_strike_to_dollar(S0)
        call_premium = black_scholes_price(S0, K, T, risk_free_rate, sigma, option_type="call")
        put_premium  = black_scholes_price(S0, K, T, risk_free_rate, sigma, option_type="put")
        total_cost = call_premium + put_premium
        call_payoff = max(ST - K, 0)
        put_payoff  = max(K - ST, 0)
        total_payoff = call_payoff + put_payoff
        pl = total_payoff - total_cost
        results.append({
            "Strategy": "Straddle",
            "Entry Date": entry_date,
            "Expiry Date": expiry_date,
            "Strike(s)": f"{K}",
            "Cost": total_cost,
            "Payoff": total_payoff,
            "P/L": pl,
            "S0": S0,
            "ST": ST,
            "Vol": sigma
        })
    elif strategy_type == "Strangle":
        pct = strangle_pct / 100.0
        K_put  = round_strike_to_dollar(S0 * (1 - pct))
        K_call = round_strike_to_dollar(S0 * (1 + pct))
        put_premium  = black_scholes_price(S0, K_put, T, risk_free_rate, sigma, option_type="put")
        call_premium = black_scholes_price(S0, K_call, T, risk_free_rate, sigma, option_type="call")
        total_cost = put_premium + call_premium
        put_payoff  = max(K_put - ST, 0)
        call_payoff = max(ST - K_call, 0)
        total_payoff = put_payoff + call_payoff
        pl = total_payoff - total_cost
        results.append({
            "Strategy": "Strangle",
            "Entry Date": entry_date,
            "Expiry Date": expiry_date,
            "Strike(s)": f"Put {K_put}, Call {K_call}",
            "Cost": total_cost,
            "Payoff": total_payoff,
            "P/L": pl,
            "S0": S0,
            "ST": ST,
            "Vol": sigma
        })
    elif strategy_type == "Iron Condor":
        pctD = ic_lower_put_pct / 100.0
        pctC = ic_outer_put_pct / 100.0
        pctA = ic_inner_call_pct / 100.0
        pctB = ic_outer_call_pct / 100.0
        K_D = round_strike_to_dollar(S0 * (1 - pctD))
        K_C = round_strike_to_dollar(S0 * (1 - pctC))
        K_A = round_strike_to_dollar(S0 * (1 + pctA))
        K_B = round_strike_to_dollar(S0 * (1 + pctB))
        cA = black_scholes_price(S0, K_A, T, risk_free_rate, sigma, option_type="call")
        cB = black_scholes_price(S0, K_B, T, risk_free_rate, sigma, option_type="call")
        pC = black_scholes_price(S0, K_C, T, risk_free_rate, sigma, option_type="put")
        pD = black_scholes_price(S0, K_D, T, risk_free_rate, sigma, option_type="put")
        net_credit = cA - cB + pC - pD
        if ST <= K_A:
            bear_call_payoff = 0
        elif K_A < ST <= K_B:
            bear_call_payoff = -(ST - K_A)
        else:
            bear_call_payoff = -(K_B - K_A)
        if ST >= K_C:
            bull_put_payoff = 0
        elif K_D <= ST < K_C:
            bull_put_payoff = -(K_C - ST)
        else:
            bull_put_payoff = -(K_C - K_D)
        total_payoff = bear_call_payoff + bull_put_payoff
        pl = net_credit + total_payoff
        results.append({
            "Strategy": "Iron Condor",
            "Entry Date": entry_date,
            "Expiry Date": expiry_date,
            "Strike(s)": f"Puts: {K_D}/{K_C}, Calls: {K_A}/{K_B}",
            "Cost": -net_credit,
            "Credit Received": net_credit,
            "Payoff": total_payoff,
            "P/L": pl,
            "S0": S0,
            "ST": ST,
            "Vol": sigma
        })

# ---------------------------------------------------
# 6. Display & Plot Results
# ---------------------------------------------------
df_results = pd.DataFrame(results)

if df_results.empty:
    st.error("No completed trades found for these parameters. Try widening your date range or adjusting DTE.")
    st.stop()

st.subheader("Backtest Results (Per Trade)")
st.dataframe(df_results)

total_pl = df_results["P/L"].sum()
avg_pl = df_results["P/L"].mean()
win_rate = (df_results["P/L"] > 0).mean()

st.markdown("### Summary Metrics")
st.write(f"**Total P/L:** ${total_pl:,.2f}")
st.write(f"**Average P/L per Trade:** ${avg_pl:,.2f}")
st.write(f"**Win Rate:** {win_rate * 100:.1f}%")

df_results = df_results.sort_values("Entry Date").reset_index(drop=True)
df_results["Cumulative P/L"] = df_results["P/L"].cumsum()

fig_equity = px.line(
    df_results, 
    x="Entry Date", 
    y="Cumulative P/L", 
    title="Equity Curve (Cumulative P/L over Time)",
    labels={"Cumulative P/L": "Cumulative P/L (USD)"}
)
st.plotly_chart(fig_equity, use_container_width=True)

fig_hist = px.histogram(
    df_results, 
    x="P/L", 
    nbins=30, 
    title="Histogram of Individual Trade P/L",
    labels={"P/L": "P/L per Trade (USD)"}
)
st.plotly_chart(fig_hist, use_container_width=True)

csv_data = df_results.to_csv(index=False)
st.download_button(
    label="Download Results as CSV",
    data=csv_data,
    file_name="backtest_results.csv",
    mime="text/csv"
) 
