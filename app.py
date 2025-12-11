# app.py (full file)
import os
import csv
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

from model_utils import (
    download_data, add_basic_indicators, trend_rule,
    add_target_big_move, train_ml_model, predict_ml, ensemble_score
)

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "logs.csv")

CATEGORIES = {
    "India (Blue-chip)": [
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "LT.NS","MARUTI.NS","BHARTIARTL.NS","HINDUNILVR.NS","ASIANPAINT.NS"
    ],
    "Penny / Speculative": [
        "IDEA.NS","YESBANK.NS","SUZLON.NS","JPASSOCIAT.NS","RPOWER.NS",
        "ZOMATO.NS","COALINDIA.NS","PNB.NS"
    ]
}

def log_run(row, header=None):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if (not file_exists) and header:
            writer.writerow(header)
        writer.writerow(row)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", categories=CATEGORIES)

@app.route("/predict", methods=["POST"])
def predict():
    category = request.form.get("category", "")
    ticker = request.form.get("ticker", "").strip()
    start = request.form.get("start", "2018-01-01")
    end = request.form.get("end", None)

    if not ticker:
        return redirect(url_for("index"))

    # download data
    try:
        df = download_data(ticker, start=start, end=end)
    except Exception as e:
        return render_template("result.html", error=str(e), ticker=ticker)

    try:
        df = add_basic_indicators(df)
    except Exception as e:
        return render_template("result.html", error="Indicator error: " + str(e), ticker=ticker)

    # Trend rule
    try:
        trend_label, trend_score, ma_slope, last_rsi, explain = trend_rule(df)
    except Exception as e:
        trend_label, trend_score, ma_slope, last_rsi, explain = "NEUTRAL", 0.0, 0.0, 0.0, f"Trend rule error: {e}"

    # Prepare ML target + train/predict
    try:
        df_ml = add_target_big_move(df)  # adds target column
        model, mean_auc, feature_cols = train_ml_model(df_ml)
        prob_up = predict_ml(model, df_ml, feature_cols)
    except Exception as e:
        mean_auc, prob_up = None, 0.0

    # Ensemble combine
    try:
        final_score, final_label, confidence = ensemble_score(prob_up, trend_score, w_ml=0.6, w_rule=0.4)
    except Exception as e:
        final_score, final_label, confidence = 0.0, "NEUTRAL", 0.5

    # build Plotly figure: candlestick + MA + volume
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", increasing_line_color='green', decreasing_line_color='red',
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma10"], mode='lines', name='MA10', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma50"], mode='lines', name='MA50', line=dict(width=1)))
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color='lightgray', yaxis='y2'))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        title=f"{ticker} — Interactive chart (candles + MA10/MA50)",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, position=1.0, range=[0, max(df["Volume"]) * 3])
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), template="plotly_white", height=700)
    fig.update_layout(autosize=True)   # ensure autosize is enabled
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})

    # logging row (extend to include ML + ensemble info)
    last_close = float(df["Close"].iloc[-1])
    n_points = len(df)
    row = [
        datetime.now().strftime("%Y-%m-%d"),
        ticker,
        round(last_close,4),
        n_points,
        start,
        end or "",
        category,
        trend_label,
        round(trend_score,4),
        round(ma_slope,6),
        round(last_rsi,2),
        round(prob_up,4) if prob_up is not None else "",
        round(mean_auc,4) if mean_auc is not None else "",
        round(final_score,4),
        final_label,
        round(confidence,4)
    ]
    log_header = [
        "date_run","ticker","last_close","n_points","start","end","category",
        "trend_label","trend_score","ma_slope","rsi","prob_up","mean_auc",
        "final_score","final_label","confidence"
    ]
    log_run(row, header=log_header)

    summary = {
        "ticker": ticker,
        "last_close": last_close,
        "n_points": n_points,
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
        "trend_label": trend_label,
        "trend_score": trend_score,
        "ma_slope": ma_slope,
        "rsi": last_rsi,
        "explain": explain,
        "prob_up": prob_up,
        "mean_auc": mean_auc,
        "final_score": final_score,
        "final_label": final_label,
        "confidence": confidence
    }

    return render_template("result.html", fig_html=fig_html, summary=summary)



@app.route("/logs")
def view_logs():
    try:
        df = pd.read_csv(LOG_FILE)
        df = df.fillna("")  # avoid NaN in HTML
        logs = df.to_dict(orient="records")
    except Exception as e:
        logs = []
        print("Error loading logs:", e)

    return render_template("logs.html", logs=logs)


if __name__ == "__main__":
    app.run(debug=True)
