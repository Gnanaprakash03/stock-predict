# model_utils.py
import pandas as pd
import yfinance as yf
import ta
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

MA_SHORT = 10
MA_LONG = 50

def download_data(ticker, start="2018-01-01", end=None):
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, group_by="column")
    if df.empty:
        raise ValueError("No data downloaded. Check ticker or internet or symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def add_basic_indicators(df):
    df = df.copy()
    df["ma10"] = df["Close"].rolling(MA_SHORT).mean()
    df["ma50"] = df["Close"].rolling(MA_LONG).mean()
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["vol10"] = df["Volume"].rolling(10).mean()
    df.dropna(inplace=True)
    return df

def add_target_big_move(df):
    """
    Add target column: 1 if next-day return > +1%, else 0
    """
    df = df.copy()
    df["next_ret"] = df["Close"].shift(-1) / df["Close"] - 1
    df["target"] = (df["next_ret"] > 0.01).astype(int)
    df.dropna(inplace=True)
    return df

# ---------------- Trend rule (as before) ----------------
def trend_rule(df: pd.DataFrame):
    """
    Return (label, score, ma_slope, rsi, explain_text)
    label in {"UP","NEUTRAL","DOWN"}
    score in [-1,1] (positive=bullish, negative=bearish)
    """
    if len(df) < MA_LONG + 6:
        return "NEUTRAL", 0.0, 0.0, float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 0.0, "Not enough data for trend rule."

    ma_s = float(df["ma10"].iloc[-1])
    ma_l = float(df["ma50"].iloc[-1])
    ma_s_prev = float(df["ma10"].iloc[-6])
    ma_slope = (ma_s - ma_s_prev) / (ma_s_prev if ma_s_prev != 0 else 1.0)
    rsi = float(df["rsi"].iloc[-1])

    trend_score = 0.0
    trend_label = "NEUTRAL"
    reasons = []

    if (ma_s > ma_l) and (ma_slope > 0.005) and (40 <= rsi <= 70):
        trend_score = 0.8
        trend_label = "UP"
        reasons.append("MA10 > MA50, positive short-MA slope, RSI neutral")
    elif (ma_s > ma_l) and (ma_slope > 0.0) and (rsi > 35):
        trend_score = 0.4
        trend_label = "UP"
        reasons.append("MA10 > MA50, mild positive slope")
    elif (ma_s < ma_l) and (ma_slope < -0.005) and (rsi < 60):
        trend_score = -0.8
        trend_label = "DOWN"
        reasons.append("MA10 < MA50, negative short-MA slope")
    elif (ma_s < ma_l) and (ma_slope < 0.0):
        trend_score = -0.4
        trend_label = "DOWN"
        reasons.append("MA10 < MA50, mild negative slope")
    else:
        trend_score = 0.0
        trend_label = "NEUTRAL"
        reasons.append("No clear MA crossover / slope signal")

    explain = "; ".join(reasons) + f"; RSI={rsi:.1f}, MA_slope={ma_slope:.5f}"
    return trend_label, float(trend_score), float(ma_slope), float(rsi), explain

# ---------------- ML functions ----------------
def train_ml_model(df, feature_cols=None, n_splits=5):
    """
    Train XGBoost using TimeSeriesSplit and return (model, mean_auc, feature_cols).
    Expects df to already have target column.
    """
    if feature_cols is None:
        feature_cols = ["ret1", "ret5", "ma10", "ma50", "vol10", "rsi", "atr"]  # 'atr' might be missing depending on data
    # Ensure features present; compute ret1/ret5/atr if not present
    df2 = df.copy()
    if "ret1" not in df2.columns:
        df2["ret1"] = df2["Close"].pct_change(1)
    if "ret5" not in df2.columns:
        df2["ret5"] = df2["Close"].pct_change(5)
    if "atr" not in df2.columns:
        try:
            df2["atr"] = ta.volatility.AverageTrueRange(high=df2["High"], low=df2["Low"], close=df2["Close"], window=14).average_true_range()
        except Exception:
            df2["atr"] = 0.0

    df2 = df2.dropna()
    X = df2[feature_cols].values
    y = df2["target"].values
    if len(y) < 50:
        # too small to properly train; return a default model trained minimally
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss")
        model.fit(X, y)
        return model, None, feature_cols

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=3, eval_metric="logloss")
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        try:
            aucs.append(roc_auc_score(yte, prob))
        except:
            pass

    final = xgb.XGBClassifier(n_estimators=200, max_depth=3, eval_metric="logloss")
    final.fit(X, y)
    mean_auc = float(np.mean(aucs)) if len(aucs) > 0 else None
    return final, mean_auc, feature_cols

def predict_ml(model, df, feature_cols):
    """
    Return the prob_up for the last row (float).
    """
    df2 = df.copy()
    # ensure features exist
    if "ret1" not in df2.columns:
        df2["ret1"] = df2["Close"].pct_change(1)
    if "ret5" not in df2.columns:
        df2["ret5"] = df2["Close"].pct_change(5)
    if "atr" not in df2.columns:
        try:
            df2["atr"] = ta.volatility.AverageTrueRange(high=df2["High"], low=df2["Low"], close=df2["Close"], window=14).average_true_range()
        except Exception:
            df2["atr"] = 0.0
    df2 = df2.dropna()
    if df2.empty:
        return 0.0
    X = df2[feature_cols].values
    prob_series = model.predict_proba(X)[:, 1]
    return float(prob_series[-1])



def ensemble_score(prob_up, trend_score, w_ml=0.6, w_rule=0.4):
    """
    Combine ML prob_up and trend_score into a final_score in [-1,1].
    prob_up: float in [0,1] (probability predicted by ML)
    trend_score: float in [-1,1] (from trend_rule)
    w_ml, w_rule: weights for ML and rule (sum not required; they will be normalized)
    Returns: (final_score, final_label, confidence)
      - final_score in [-1,1]
      - final_label in {"UP","NEUTRAL","DOWN"}
      - confidence in [0,1] (linear mapping)
    """
    # sanitize inputs
    if prob_up is None:
        prob_up = 0.0
    if trend_score is None:
        trend_score = 0.0

    # map prob_up from [0,1] to [-1,1]
    ml_part = 2.0 * float(prob_up) - 1.0

    # normalize weights to sum to 1
    total = float(w_ml) + float(w_rule)
    if total == 0:
        w_ml_n = 0.6
        w_rule_n = 0.4
    else:
        w_ml_n = float(w_ml) / total
        w_rule_n = float(w_rule) / total

    final_score = w_ml_n * ml_part + w_rule_n * float(trend_score)
    # clamp to [-1,1]
    final_score = max(-1.0, min(1.0, final_score))

    # thresholds for label
    if final_score >= 0.4:
        label = "UP"
    elif final_score <= -0.4:
        label = "DOWN"
    else:
        label = "NEUTRAL"

    # confidence: map [-1,1] -> [0,1]
    confidence = (final_score + 1.0) / 2.0
    return float(final_score), label, float(confidence)