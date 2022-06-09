import pandas as pd
import yfinance as yf
import datetime
from datetime import date, datetime, timedelta
today = date.today()

# d1 = today.strftime("%Y-%m-%d")
d1 = datetime.now()
end_date = d1
d2 = datetime.now() - timedelta(minutes = 59 * 24 * 60)
# d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('BNB-USD', 
                      start=start_date, 
                      end=end_date, 
                      progress=False,
                      interval='5m')
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)

# print(data.tail())
# print(data.shape)

# visualize the change in bnb prices till today by using a candlestick chart

# import plotly.graph_objects as go
# figure = go.Figure(data=[go.Candlestick(x=data["Date"],
#                                         open=data["Open"], 
#                                         high=data["High"],
#                                         low=data["Low"], 
#                                         close=data["Close"])])
# figure.update_layout(title = "Bitcoin Price Analysis", 
#                      xaxis_rangeslider_visible=False)
# figure.show()

# Viewing the correlation betwween the "Close" column and the other columns
# correlation = data.corr()
# print(correlation["Close"].sort_values(ascending=False))

# AutoTS is a Tme Series library for Python
from autots import AutoTS

model = AutoTS(forecast_length=3, frequency='infer', ensemble='simple')

# Creating the model
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

# The name of the best model
print(f"BEST MODEL {model}")

# prediction anf forecasting
prediction = model.predict()
forecast = prediction.forecast

model_results = model.results()

validation = model.results("validation")

print(forecast)

print(validation)


