import streamlit as st
from stock_price import fetch_stock_data, plot_comparison, plot_peaks

def main():
    st.title("Stock Comparison and Peak Analysis")
    api_key = st.text_input("Enter your Alpha Vantage API key:", type="password")
    option = st.selectbox("Choose an option:", ["Compare two stocks", "See peaks for a single stock"])

    if option == "Compare two stocks":
        stock1 = st.text_input("Enter the first stock symbol:")
        stock2 = st.text_input("Enter the second stock symbol:")
        if st.button("Compare"):
            try:
                df1 = fetch_stock_data(stock1, api_key)
                df2 = fetch_stock_data(stock2, api_key)
                plot_comparison(stock1, stock2, df1, df2)
            except KeyError as e:
                st.error(e)
    elif option == "See peaks for a single stock":
        stock = st.text_input("Enter the stock symbol:")
        if st.button("Show Peaks"):
            try:
                df = fetch_stock_data(stock, api_key)
                plot_peaks(stock, df)
            except KeyError as e:
                st.error(e)

if __name__ == "__main__":
    main()