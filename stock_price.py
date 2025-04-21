import requests
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks
import streamlit as st

def fetch_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    print(data)

    if 'Time Series (Daily)' not in data:
        raise KeyError(f"Data for {symbol} not found. Please check the symbol and try again.")
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    df = df[df.index >= six_months_ago]
    return df

def plot_comparison(stock1, stock2, df1, df2):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df1['4. close'], label=f'{stock1} Close Price')
    ax.plot(df2['4. close'], label=f'{stock2} Close Price')
    ax.set_title(f'Comparison of {stock1} and {stock2} Stock Prices in the Last 6 Months')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

def plot_peaks(symbol, df):
    close_prices = df['4. close']
    peaks, _ = find_peaks(close_prices)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(close_prices, label='Close Price')
    ax.plot(close_prices.index[peaks], close_prices.iloc[peaks], 'x', label='Peaks')
    ax.set_title(f'{symbol} Stock Price and Peaks in the Last 6 Months')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)
    st.write(f'Number of peaks in the last 6 months for {symbol}: {len(peaks)}')

def main():
    api_key = 'R7S4BF8N5GX30QF8'
    option = input("Enter '1' to compare two stocks or '2' to see peaks for a single stock: ")

    if option == '1':
        stock1 = input("Enter the first stock symbol: ")
        stock2 = input("Enter the second stock symbol: ")
        try:
            df1 = fetch_stock_data(stock1, api_key)
            df2 = fetch_stock_data(stock2, api_key)
            plot_comparison(stock1, stock2, df1, df2)
        except KeyError as e:
            print(e)
    elif option == '2':
        stock = input("Enter the stock symbol: ")
        try:
            df = fetch_stock_data(stock, api_key)
            plot_peaks(stock, df)
        except KeyError as e:
            print(e)
    else:
        print("Invalid option. Please enter '1' or '2'.")

if __name__ == "__main__":
    main()