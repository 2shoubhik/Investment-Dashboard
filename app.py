import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shoubhik's Investment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1c2333;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .metric-label { color: #a0aec0; font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
    .metric-value { color: #e2e8f0; font-size: 26px; font-weight: 700; margin-top: 4px; }
    .metric-delta-pos { color: #48bb78; font-size: 13px; }
    .metric-delta-neg { color: #fc8181; font-size: 13px; }
    .section-header {
        color: #e2e8f0;
        font-size: 18px;
        font-weight: 700;
        padding: 12px 0 6px 0;
        border-bottom: 2px solid #2d3748;
        margin-bottom: 16px;
    }
    .sentiment-positive { color: #48bb78; font-weight: 600; }
    .sentiment-negative { color: #fc8181; font-weight: 600; }
    .sentiment-neutral  { color: #a0aec0; font-weight: 600; }
    .ticker-pill {
        display: inline-block;
        background: #2d3748;
        color: #90cdf4;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 13px;
        font-weight: 700;
        margin: 2px;
    }
    div[data-testid="stSidebar"] { background-color: #141925; }
    div[data-testid="metric-container"] { background: #1c2333; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.045  # ~current 3-month T-bill rate
DEFAULT_TICKERS = ["SNPS", "PLTR", "APP", "BTC-USD", "MFC", "NVDA", "VRT", "AAPL", "UNH", "NVO"]
PERIODS = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}

# ─── DATA FETCHING ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_price_history(tickers: list, period: str) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers."""
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = tickers
        return prices.dropna(how="all")
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_ticker_info(ticker: str) -> dict:
    """Fetch fundamental data for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name":         info.get("longName", ticker),
            "sector":       info.get("sector", "—"),
            "market_cap":   info.get("marketCap"),
            "pe_ratio":     info.get("trailingPE"),
            "pb_ratio":     info.get("priceToBook"),
            "ev_ebitda":    info.get("enterpriseToEbitda"),
            "price":        info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high":     info.get("fiftyTwoWeekHigh"),
            "52w_low":      info.get("fiftyTwoWeekLow"),
            "currency":     info.get("currency", "USD"),
        }
    except Exception:
        return {}

@st.cache_data(ttl=600)
def fetch_reddit_sentiment(ticker: str) -> list:
    """Fetch recent Reddit mentions via public JSON endpoint and score sentiment."""
    try:
        results = []
        subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket', 'CanadianInvestor']
        
        headers = {'User-Agent': 'investment-dashboard/1.0'}
        
        for subreddit in subreddits:
            # Search via Reddit's public JSON API
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': f"{ticker} OR ${ticker}",
                'restrict_sr': 'on',
                'sort': 'relevance',
                'limit': 5,
                't': 'week'
            }
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=5)
                if response.status_code != 200:
                    continue
                    
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                
                for post in posts:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    if not title or len(title) < 10:
                        continue
                    
                    # Sentiment analysis
                    blob = TextBlob(title)
                    polarity = blob.sentiment.polarity
                    
                    if polarity > 0.1:
                        label, css = "Bullish", "sentiment-positive"
                    elif polarity < -0.1:
                        label, css = "Bearish", "sentiment-negative"
                    else:
                        label, css = "Neutral", "sentiment-neutral"
                    
                    results.append({
                        "headline": title[:150],
                        "polarity": polarity,
                        "label": label,
                        "css": css,
                        "url": f"https://reddit.com{post_data.get('permalink', '')}",
                        "published": datetime.fromtimestamp(post_data.get('created_utc', 0)).strftime("%b %d"),
                        "subreddit": subreddit,
                        "score": post_data.get('score', 0),
                        "comments": post_data.get('num_comments', 0)
                    })
            except Exception:
                continue
        
        # Sort by Reddit score and remove duplicates
        seen_titles = set()
        unique_results = []
        for item in sorted(results, key=lambda x: x['score'], reverse=True):
            if item['headline'] not in seen_titles:
                seen_titles.add(item['headline'])
                unique_results.append(item)
        
        return unique_results[:10]
        
    except Exception as e:
        return []

# ─── RISK CALCULATIONS ───────────────────────────────────────────────────────────
def compute_risk_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of risk metrics for each ticker."""
    returns = prices.pct_change().dropna()
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 10:
            continue
        ann_return   = (1 + r.mean()) ** 252 - 1
        ann_vol      = r.std() * np.sqrt(252)
        sharpe       = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
        # Max drawdown
        cum = (1 + r).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        max_dd   = drawdown.min()
        # Sortino (downside deviation)
        downside = r[r < 0].std() * np.sqrt(252)
        sortino  = (ann_return - RISK_FREE_RATE) / downside if downside > 0 else np.nan

        rows.append({
            "Ticker":          col,
            "Ann. Return":     ann_return,
            "Ann. Volatility": ann_vol,
            "Sharpe Ratio":    sharpe,
            "Sortino Ratio":   sortino,
            "Max Drawdown":    max_dd,
        })
    return pd.DataFrame(rows).set_index("Ticker")

def compute_normalised_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalise prices to 100 at start for comparison."""
    if prices.empty or len(prices) == 0:
        return pd.DataFrame()
    return (prices / prices.iloc[0]) * 100

# ─── CHART HELPERS ───────────────────────────────────────────────────────────────
CHART_COLORS = ["#4299e1","#48bb78","#ed8936","#9f7aea","#f56565","#38b2ac","#ecc94b","#667eea"]

def price_chart(norm: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], name=col,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d %Y}}<br>Index: %{{y:.1f}}<extra></extra>"
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=0, r=0, t=30, b=0), height=380,
        yaxis_title="Indexed to 100", xaxis_title=""
    )
    return fig

def drawdown_chart(prices: pd.DataFrame) -> go.Figure:
    returns = prices.pct_change().dropna()
    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        r = returns[col].dropna()
        cum = (1 + r).cumprod()
        roll_max = cum.cummax()
        dd = ((cum - roll_max) / roll_max) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=col, fill="tozeroy",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
            fillcolor=f"rgba({int(CHART_COLORS[i % len(CHART_COLORS)][1:3], 16)}, {int(CHART_COLORS[i % len(CHART_COLORS)][3:5], 16)}, {int(CHART_COLORS[i % len(CHART_COLORS)][5:7], 16)}, 0.15)",
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d}}<br>Drawdown: %{{y:.1f}}%<extra></extra>"
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=0, r=0, t=30, b=0), height=300,
        yaxis_title="Drawdown (%)", xaxis_title=""
    )
    return fig

def correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    corr = prices.pct_change().dropna().corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{x} / %{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=10, b=0), height=350
    )
    return fig

def risk_scatter(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (ticker, row) in enumerate(metrics.iterrows()):
        fig.add_trace(go.Scatter(
            x=[row["Ann. Volatility"] * 100],
            y=[row["Ann. Return"] * 100],
            mode="markers+text",
            name=ticker,
            text=[ticker],
            textposition="top center",
            marker=dict(size=14, color=CHART_COLORS[i % len(CHART_COLORS)]),
            hovertemplate=f"<b>{ticker}</b><br>Volatility: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#4a5568")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Annualised Volatility (%)", yaxis_title="Annualised Return (%)",
        showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=350
    )
    return fig

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    tickers_input = st.text_area(
        "Watchlist Tickers",
        value="\n".join(DEFAULT_TICKERS),
        height=160,
        help="One ticker per line. Use Yahoo Finance symbols (e.g. BTC-USD for Bitcoin, BAM for Brookfield)."
    )
    tickers = [t.strip().upper() for t in tickers_input.strip().split("\n") if t.strip()]

    period_label = st.selectbox("Time Period", list(PERIODS.keys()), index=3)
    period = PERIODS[period_label]

    st.markdown("---")
    st.markdown("### 📌 Benchmark")
    benchmark = st.text_input("Benchmark Ticker", value="SPY", help="Compared against your watchlist in the returns chart.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5568'>Data via Yahoo Finance · Sentiment via TextBlob · "
        "Built with Streamlit & Plotly</small>", unsafe_allow_html=True
    )

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────────
st.markdown("# Shoubhik's Investment Dashboard")
st.markdown(f"<small style='color:#4a5568'>Last updated: {datetime.now().strftime('%b %d, %Y  %H:%M')}</small>", unsafe_allow_html=True)
st.markdown("---")

if not tickers:
    st.warning("Add at least one ticker in the sidebar to get started.")
    st.stop()

# Fetch data
all_tickers = list(set(tickers + [benchmark]))
with st.spinner("Fetching market data..."):
    prices_all = fetch_price_history(all_tickers, period)

if prices_all.empty:
    st.error("Could not fetch price data. Check your tickers and internet connection.")
    st.stop()

# Separate watchlist prices from benchmark
available = [t for t in tickers if t in prices_all.columns]
prices     = prices_all[available].dropna(how="all")
bench_prices = prices_all[[benchmark]] if benchmark in prices_all.columns else pd.DataFrame()

# ─── TAB LAYOUT ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Overview", "⚠️  Risk & Returns", "💰  Valuation", "💬  Reddit Sentiment"])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    # Quick metrics row
    st.markdown('<div class="section-header">Watchlist Snapshot</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(available), 6))
    for i, ticker in enumerate(available[:6]):
        info = fetch_ticker_info(ticker)
        price = info.get("price")
        col_prices = prices[ticker].dropna()
        if len(col_prices) >= 2:
            chg = (col_prices.iloc[-1] / col_prices.iloc[0] - 1) * 100
            delta_str = f"{'▲' if chg >= 0 else '▼'} {abs(chg):.1f}% ({period_label})"
            delta_css = "metric-delta-pos" if chg >= 0 else "metric-delta-neg"
        else:
            delta_str, delta_css = "—", "metric-delta-neutral"
        price_str = f"${price:,.2f}" if price else "—"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{ticker}</div>
                <div class="metric-value">{price_str}</div>
                <div class="{delta_css}">{delta_str}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Normalised price chart
    st.markdown('<div class="section-header">Price Performance (Indexed to 100)</div>', unsafe_allow_html=True)
    prices_clean = prices.dropna()
    if len(prices_clean) < 2:
        st.warning("Not enough price data for the selected period. Try a longer time range.")
        st.stop()
    norm = compute_normalised_returns(prices_clean)

    # Add benchmark if available
    if not bench_prices.empty:
        bench_norm = compute_normalised_returns(bench_prices.dropna())
        bench_norm.columns = [f"{benchmark} (benchmark)"]
        combined_norm = pd.concat([norm, bench_norm], axis=1).dropna()
    else:
        combined_norm = norm

    st.plotly_chart(price_chart(combined_norm), use_container_width=True)

    # Drawdown
    st.markdown('<div class="section-header">Drawdown from Peak</div>', unsafe_allow_html=True)
    st.plotly_chart(drawdown_chart(prices.dropna()), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK & RETURNS
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:
    metrics = compute_risk_metrics(prices.dropna())

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Risk / Return Scatter</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_scatter(metrics), use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        if len(available) > 1:
            st.plotly_chart(correlation_heatmap(prices.dropna()), use_container_width=True)
        else:
            st.info("Add more tickers to see correlations.")

    st.markdown('<div class="section-header">Risk Metrics Table</div>', unsafe_allow_html=True)
    display = metrics.copy()
    display["Ann. Return"]     = display["Ann. Return"].map("{:.1%}".format)
    display["Ann. Volatility"] = display["Ann. Volatility"].map("{:.1%}".format)
    display["Sharpe Ratio"]    = display["Sharpe Ratio"].map("{:.2f}".format)
    display["Sortino Ratio"]   = display["Sortino Ratio"].map("{:.2f}".format)
    display["Max Drawdown"]    = display["Max Drawdown"].map("{:.1%}".format)
    st.dataframe(display, use_container_width=True)

    st.markdown(
        "<small style='color:#4a5568'>Sharpe & Sortino use annualised figures vs. "
        f"{RISK_FREE_RATE*100:.1f}% risk-free rate. Based on {period_label} of daily returns.</small>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — VALUATION
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Valuation Multiples</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#4a5568'>Note: Crypto assets will not have valuation multiples.</small><br>", unsafe_allow_html=True)

    val_rows = []
    val_progress = st.progress(0, text="Loading fundamentals...")
    for i, ticker in enumerate(available):
        info = fetch_ticker_info(ticker)
        val_rows.append({
            "Ticker":     ticker,
            "Name":       info.get("name", "—")[:35],
            "Sector":     info.get("sector", "—"),
            "Price":      f"${info['price']:,.2f}" if info.get("price") else "—",
            "Mkt Cap":    f"${info['market_cap']/1e9:.1f}B" if info.get("market_cap") else "—",
            "P/E":        f"{info['pe_ratio']:.1f}x" if info.get("pe_ratio") else "—",
            "P/B":        f"{info['pb_ratio']:.1f}x" if info.get("pb_ratio") else "—",
            "EV/EBITDA":  f"{info['ev_ebitda']:.1f}x" if info.get("ev_ebitda") else "—",
            "52W High":   f"${info['52w_high']:,.2f}" if info.get("52w_high") else "—",
            "52W Low":    f"${info['52w_low']:,.2f}" if info.get("52w_low") else "—",
        })
        val_progress.progress((i + 1) / len(available), text=f"Loading {ticker}...")

    val_progress.empty()
    val_df = pd.DataFrame(val_rows).set_index("Ticker")
    st.dataframe(val_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 4 — NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Reddit Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#4a5568'>Sourced from r/stocks, r/investing, r/wallstreetbets, r/StockMarket, and r/CanadianInvestor. Sentiment scored via TextBlob NLP.</small><br>", unsafe_allow_html=True)

    selected_ticker = st.selectbox("Select ticker for Reddit mentions", available)

    with st.spinner(f"Fetching Reddit posts for {selected_ticker}..."):
        news_items = fetch_reddit_sentiment(selected_ticker)

    if not news_items:
        st.info(f"No recent Reddit mentions found for {selected_ticker} in the past week.")
    else:
        # Aggregate sentiment
        polarities = [n["polarity"] for n in news_items]
        avg_pol = np.mean(polarities)
        pos_count = sum(1 for p in polarities if p > 0.1)
        neg_count = sum(1 for p in polarities if p < -0.1)
        neu_count = len(polarities) - pos_count - neg_count

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Avg. Sentiment Score", f"{avg_pol:+.3f}", help="-1 (very bearish) to +1 (very bullish)")
        with m2:
            st.metric("🟢 Bullish Posts", pos_count)
        with m3:
            st.metric("⚪ Neutral Posts", neu_count)
        with m4:
            st.metric("🔴 Bearish Posts", neg_count)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sentiment bar chart
        fig_sent = go.Figure(go.Bar(
            x=[n["headline"][:55] + "…" if len(n["headline"]) > 55 else n["headline"] for n in news_items],
            y=[n["polarity"] for n in news_items],
            marker_color=["#48bb78" if n["polarity"] > 0.1 else "#fc8181" if n["polarity"] < -0.1 else "#a0aec0" for n in news_items],
            hovertext=[n["headline"] for n in news_items],
            hovertemplate="<b>%{hovertext}</b><br>Polarity: %{y:.3f}<extra></extra>"
        ))
        fig_sent.add_hline(y=0, line_dash="dot", line_color="#4a5568")
        fig_sent.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            xaxis_tickangle=-30, yaxis_title="Sentiment Polarity"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        # Reddit posts table
        st.markdown('<div class="section-header">Recent Reddit Mentions</div>', unsafe_allow_html=True)
        for item in news_items:
            st.markdown(
                f"<span class='{item['css']}'>{item['label']}</span> &nbsp;"
                f"<small style='color:#4a5568'>{item['published']} · r/{item['subreddit']} · ↑{item['score']} · {item['comments']} comments</small>&nbsp; "
                f"<a href='{item['url']}' target='_blank' style='color:#90cdf4; text-decoration:none'>{item['headline']}</a>",
                unsafe_allow_html=True
            )
            st.markdown("<hr style='border-color:#2d3748; margin: 6px 0'>", unsafe_allow_html=True)
