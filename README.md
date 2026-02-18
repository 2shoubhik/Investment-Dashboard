# 📈 Investment Dashboard

A personal portfolio analytics and stock screening dashboard built with Python and Streamlit. Tracks equities and crypto across risk, valuation, and news sentiment dimensions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Watchlist tracking** — monitor any mix of equities, ETFs, and crypto via Yahoo Finance symbols
- **Performance charts** — normalised price comparison indexed to 100, with a configurable benchmark (default: SPY)
- **Drawdown analysis** — visualise peak-to-trough drawdowns across your watchlist
- **Risk metrics** — annualised return, volatility, Sharpe ratio, Sortino ratio, and max drawdown
- **Correlation heatmap** — understand diversification across your holdings
- **Risk/return scatter** — quickly identify your best and worst risk-adjusted positions
- **Valuation multiples** — P/E, P/B, EV/EBITDA, market cap, and 52-week range per ticker
- **News sentiment** — recent headlines scored via NLP (TextBlob), with aggregate sentiment summary

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/investment-dashboard.git
cd investment-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

- **Sidebar → Watchlist Tickers**: enter one ticker per line using Yahoo Finance symbols
  - Stocks: `AAPL`, `BAM`, `TD`
  - Crypto: `BTC-USD`, `ETH-USD`
  - ETFs: `SPY`, `QQQ`, `XIC.TO`
  - TSX stocks: append `.TO` (e.g. `BAM.A.TO`)
- **Sidebar → Time Period**: select 1 month to 2 years of history
- **Sidebar → Benchmark**: compare your watchlist against any index or ticker (default: SPY)

---

## Project Structure

```
investment-dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [yfinance](https://github.com/ranaroussi/yfinance) | Market data (Yahoo Finance API) |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [pandas / numpy](https://pandas.pydata.org) | Data manipulation & risk calculations |
| [TextBlob](https://textblob.readthedocs.io) | NLP sentiment scoring |

---

## Methodology Notes

- **Sharpe & Sortino Ratios** are annualised using daily returns × √252, benchmarked against a 4.5% risk-free rate (approximate current 3-month T-bill)
- **Sentiment scoring** uses TextBlob's pattern-based NLP on headline text — a simple but interpretable baseline; positive > 0.1, negative < -0.1
- **Data** is sourced from Yahoo Finance via `yfinance` and cached for 5 minutes to avoid rate limits
- Crypto assets will not have valuation multiples (P/E, EV/EBITDA etc.)

---

## Roadmap

- [ ] Portfolio weight optimisation (mean-variance / max Sharpe)
- [ ] Earnings calendar integration
- [ ] Export to PDF / CSV
- [ ] Email alerts on drawdown thresholds

---

## Disclaimer

This tool is for personal research and informational purposes only. Nothing here constitutes financial advice.

---

*Built by Shoubhik Bhattacharya*
