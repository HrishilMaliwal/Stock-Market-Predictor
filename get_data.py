import yfinance as yf
import pandas as pd
ticker='xyz'


df=pd.read_html('https://in.finance.yahoo.com/quote/%5ENSEI/components?p=%5ENSEI&.tsrc=fin-srch')[0]
tickers=df.Symbol.to_list()
print(tickers)
while ticker not in tickers:
  ticker=input("choose one from the list:")

def get_data(ticktick):
    df1=yf.download(ticker)
    print(df1)
    #return(df1)
get_data(ticker)