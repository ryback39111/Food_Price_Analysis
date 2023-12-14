

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.statespace.sarimax import SARIMAX
 

df = pd.read_csv("eda_processed_data.csv")


df['Date'] = pd.to_datetime(df['date'])


def sarima_forecast(series, order, seasonal_order, steps):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()


app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Food Price Analysis"),
    
    # Dropdown for selecting country
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in df['country'].unique()],
        value=df['country'].unique()[0],
        multi=False
    ),
    
    # Time series plot
    dcc.Graph(id='time-series-plot'),
])
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_plot(selected_country):
    filtered_df = df[df['country'] == selected_country]

    # SARIMA Forecasting
    steps = 5  # Number of steps to forecast
    order = (1, 1, 1)  # Specify the SARIMA order (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # Specify the seasonal order (P, D, Q, S)

    # Forecast future values
    forecast, confidence_interval = sarima_forecast(filtered_df['Close'], order, seasonal_order, steps)

    # Update the dataframe with forecast values
    forecast_dates = pd.date_range(start=filtered_df['Date'].max(), periods=steps + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast, 'Lower_CI': confidence_interval.iloc[:, 0].values, 'Upper_CI': confidence_interval.iloc[:, 1].values})
    filtered_df = pd.concat([filtered_df, forecast_df], ignore_index=True)

    # Create an interactive time series plot
    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Close'],
        mode='lines+markers',
        name='Actual Prices'
    ))

    # Forecasted values
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Forecast'],
        mode='lines+markers',
        name='Forecasted Prices'
    ))

    # Confidence interval shading
    upper_lower_dates = pd.to_datetime(list(filtered_df['Date']) + list(filtered_df['Date'][::-1]))
    upper_lower_values = list(filtered_df['Upper_CI']) + list(filtered_df['Lower_CI'][::-1])

    fig.add_trace(go.Scatter(
        x=upper_lower_dates,
        y=upper_lower_values,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
 

    # Additional customization can be done here
    fig.update_layout(title=f'Time Series Plot for {selected_country} with Forecast', template='plotly_dark')

    return fig

# Run the app
if __name__ == '__main__':
    application = app.server
    application.run(debug=False)