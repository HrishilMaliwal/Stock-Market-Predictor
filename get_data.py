import yfinance as yf
import pandas as pd
df=pd.read_html('https://in.finance.yahoo.com/quote/%5ENSEI/components?p=%5ENSEI&.tsrc=fin-srch')[0]
tickers=df.Symbol.to_list()
print(tickers)
ticker='xyz'
while ticker not in tickers:
  ticker=input("choose one from the list:")
df=yf.download(ticker)
print(df)