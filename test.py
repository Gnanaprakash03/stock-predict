import yfinance as yf
print(yf.__version__)
print("download rows:", yf.download("HDFCBANK.NS", period="1mo", interval="1d", progress=False, threads=False).shape)
