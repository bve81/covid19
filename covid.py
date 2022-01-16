import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc
from dash import html
import dash_daq as daq
from plotly.subplots import make_subplots
from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from datetime import datetime, timedelta

pio.templates.default = "ggplot2"
mapbox_access_token = 'pk.eyJ1IjoiYnZlODEiLCJhIjoiY2s4c2QzeDJ6MGF4NzNlcGpmZ2pnajBpaSJ9.SqJSTzdrMoCl_upfZgC2cA'
# __________________________________yesterday and tomorrow

yesterday = (datetime.now() - timedelta(1)).strftime('%-m/%-d/%y')
yestardaytoday = (datetime.now() - timedelta(1)).strftime('%-m_%-d_%y')
daysbefore = (datetime.now() - timedelta(2)).strftime('%-m_%-d_%y')
dayexcept = (datetime.now() - timedelta(1)).strftime('%-m_%-d_%y')  # exception in case if current date are not in
# datasource yet
recoverydates = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
now = (datetime.now()).strftime('%-m/%-d/%y')  # dates for AI forecast
nextweek = (datetime.now() + timedelta(7)).strftime('%-m/%-d/%y')  # dates for AI forecast
lastweek = (datetime.now() - timedelta(8)).strftime('%-m/%-d/%y')  # dates for AI forecast

# ----------------------------------data load from source to pandas
df = pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                 r'/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df.columns = [column.replace("/", "_") for column in df.columns]

deathdata = pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                        r'/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
deathdata.columns = [column.replace("/", "_") for column in deathdata.columns]

peoplerecovered = pd.read_csv(r'https://covid.observer/ru/ru-covid.observer.csv?2022-01-12')
ddd = peoplerecovered.reindex(index = peoplerecovered.index[::-1])
ddd.reset_index(inplace = True, drop = True)


mapdata = pd.read_csv('city.csv')
currentdate = yestardaytoday

# -----------------------------------focus on Russia
df = df.query('Country_Region == "Russia"')
deathdata = deathdata.query('Country_Region == "Russia"')
# recoverydata = recoverydata.query('Country_Region == "Russia"')
# -----------------------------------Pie charts
if dayexcept not in df.columns:
    currentdate = daysbefore
    yesterday = (datetime.now() - timedelta(2)).strftime('%-m/%-d/%y')

labels = ['Умерло', 'Вылечилось']  # labels for pie chart
labelst = ['Активных']

datelist = pd.date_range(start = '1/22/2020', end = yesterday, tz = None).tolist()  # List of dates
datelistFig10 = pd.date_range(start = '3/16/2020', end = yesterday, tz = None).tolist()
datelistAI = pd.date_range(start = lastweek, end = yesterday,
                           tz = None).tolist()  # List of dates for AI forecat scatter
peoplerecovered.head()
# -----------------------------------Value from tables

for value in deathdata[daysbefore]:
    dd = value
for value in deathdata[currentdate]:
    d1 = value

d2 = peoplerecovered["Recovered cases"]
for value in df[currentdate]:
    d3 = value


deathdata2 = deathdata.loc[:, '1_22_20': currentdate]

for i in deathdata2.columns:
    for value in deathdata2.values:
        dt = value
        y = {'col': dt}
        dy = pd.DataFrame(data = y)

for value in dy['col']:
        dyd = dy['col'] - dy['col'].shift(1)
print(dyd)

for i in df.columns:
    for value in df.values:
        d = value


active = ddd["Confirmed cases"] - ddd["Recovered cases"] - ddd["Fatal cases"]
lastval=active.iloc[-1]


df2 = df.loc[:, '1_22_20': currentdate].diff(axis = 1)
for i in df2.columns:
    for value in df2.values:
        cc = value
df3 = df.loc[:, (datetime.now() - timedelta(8)).strftime('%-m_%-d_%y'): currentdate].diff(axis = 1)
for i in df3.columns:
    for value in df3.values:
        ccAI = value
ccAI02 = dyd.iloc[[*range(0), *range(-8, 0)]]


# AI etna Forecast
# Read the data
hld = pd.DataFrame({
      'holiday': 'newyear',
      'ds': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-12-31', '2022-01-01',
                            '2022-01-02','2022-01-03','2022-01-04','2022-01-05','2022-01-06',
                            '2022-01-07','2022-01-08','2022-01-09','2022-01-19']),
      'lower_window': -3,
      'upper_window': 2,
    })
dataAI = { 'dates': datelist, 'values': cc }
original_df = pd.DataFrame(data = dataAI)
original_df["timestamp"] = pd.to_datetime(original_df["dates"])
original_df["target"] = original_df["values"]
original_df.drop(columns = ["dates", "values"], inplace = True)
original_df["segment"] = "main"
original_df.head()


# Create a TSDataset
df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq = "D")

# Choose a horizon
HORIZON = 8

# Fit the pipeline
pipeline = Pipeline(model = ProphetModel(
                                          changepoints=['2022-01-10'], changepoint_range=0.2,
                                          holidays = hld, holidays_prior_scale=10.0,
                                         ), horizon = HORIZON)
pipeline.fit(ts)
#changepoints=['2022-01-10'], changepoint_range=0.2 daily_seasonality = True, seasonality_mode='multiplicative',yearly_seasonality=True, seasonality_prior_scale=10.0,
# Make the forecast
forecast_ts = pipeline.forecast()
dfs = forecast_ts.to_pandas(flatten = True)
print(dfs)
# AI forecast of ded per day
dataDYD ={'dates': datelist, 'values': dyd}
deathdata_df = pd.DataFrame(data = dataDYD)
deathdata_df["timestamp"] = pd.to_datetime(deathdata_df["dates"])
deathdata_df["target"] = deathdata_df["values"]
deathdata_df.drop(columns = ["dates", "values"], inplace = True)
deathdata_df["segment"] = "main"
deathdata_df.head()
df02 = TSDataset.to_dataset(deathdata_df)
ts02 = TSDataset(df02, freq = "D")
pipeline02 = Pipeline(model = ProphetModel( yearly_seasonality=True), horizon = HORIZON)
pipeline02.fit(ts02)
forecast_ts02 = pipeline02.forecast()
dfs02 = forecast_ts02.to_pandas(flatten = True)

print(dfs02)
# ________________________________________________________________
recvsdead = [d1, d2[0]]  # Values for Pie charts
overall = active  # Values for Pie charts

# -----------------------Pie charts

fig3 = go.Figure(data = [go.Pie(labels = labels, values = recvsdead)])
fig3.update_traces(textposition = 'inside', textinfo = 'value+label')
fig3.update_layout(title_text = "Количество умерших vs вылечившихся")


aidatelist = pd.date_range(start = now, end = nextweek, tz = None).tolist()

# -------------------------Bar charts

fig11 = go.Figure()
fig11.update_layout(title_text = 'Количество инфицированных по дням')
fig11.add_trace(go.Bar(name = 'Daily infected', x = datelist, y = cc))
fig11.add_trace(go.Scatter(x = datelist, y = cc,
                           mode = 'lines+markers',
                           name = 'Текущий тренд'))
figDYD = go.Figure()
figDYD.update_layout(title_text = 'Количество летальных исходов по дням')
figDYD.add_trace(go.Bar(name = 'Летальных случаев за день', x = datelist, y = dyd))
figDYD.add_trace(go.Scatter(x = datelist, y = dyd,
                           mode = 'lines+markers',
                           name = 'Текущий тренд'))
fig9 = go.Figure()
fig9.add_trace(go.Bar(name = 'Total infected', x = datelist, y = d[4:]))
# Change the bar mode
fig9.update_layout(barmode = 'stack')
fig9.update_layout(title_text = 'график роста количества инфицированных')
fig9.add_trace(go.Scatter(x = datelist, y = d[4:],
                          mode = 'lines+markers',
                          name = 'Текущий тренд'))

figAI = make_subplots(rows = 1, cols = 1, specs = [[{ }]], shared_xaxes = True,
                      shared_yaxes = False, vertical_spacing = 0.001)
#
figAI.update_layout(title_text = 'Прогноз заболеваемости с использованием AI на следующие 8 дней')
figAI.add_trace(go.Bar(name = 'Daily infected', x = datelistAI, y = ccAI), 1, 1)
figAI.add_trace(go.Scatter(x = datelistAI, y = ccAI,
                           mode = 'lines+markers',
                           name = 'Текущий тренд'), 1, 1)
figAI.add_trace(go.Scatter(x = aidatelist, y = dfs["target"],
                           mode = 'lines+markers',
                           name = 'Прогноз от AI'), 1, 1)

figAI02 = make_subplots(rows = 1, cols = 1, specs = [[{ }]], shared_xaxes = True,
                      shared_yaxes = False, vertical_spacing = 0.001)
#
figAI02.update_layout(title_text = 'Прогноз смертности с использованием AI на следующие 8 дней')
figAI02.add_trace(go.Bar(name = 'Умерло зв день', x = datelistAI, y = ccAI02), 1, 1)
figAI02.add_trace(go.Scatter(x = datelistAI, y = ccAI02,
                           mode = 'lines+markers',
                           name = 'Текущий тренд'), 1, 1)
figAI02.add_trace(go.Scatter(x = aidatelist, y = dfs02["target"],
                           mode = 'lines+markers',
                           name = 'Прогноз от AI'), 1, 1)

fig10 = go.Figure(data = [
    go.Bar(name = 'Все инфицированные', x = datelist, y = d[4:], width = [15], text = d[4:], textposition = 'auto'),
    go.Bar(name = 'Текущие (активные) случаи заражения', x = datelistFig10, y = active, width = [15], text = active,
           textposition = 'auto')
])
# Change the bar mode
fig10.update_layout(barmode = 'group')
fig10.update_layout(title_text = 'Общее количество заражений в сравнении с активным количеством случаев')
fig10.add_trace(go.Scatter(x = datelist, y = d[4:],
                           mode = 'lines+markers',
                           name = 'Текущий тренд'))
fig10.add_trace(go.Scatter(x = datelistFig10, y = active,
                           mode = 'lines+markers',
                           name = 'Текущий тренд'))

figDD = go.Figure()
figDD.update_layout(title_text = 'Количество летальных исходов на общее количество заражений')
figDD.add_trace(go.Bar(x = datelist, y = d[4:],
                       #                            mode = 'lines+markers',
                       name = 'Общее кол-во инфицированных'
                       ))
figDD.add_trace(go.Bar(x = datelist, y = dt,  # ,
                       #                            mode = 'lines+markers',
                       name = 'Общее кол-во летальных исходов'
                       ))
figDD.update_layout(barmode = 'relative')


for i in deathdata.columns:
    for value in deathdata.values:
        d = value

# -----------------------------Map charts
# set the geo=spatial data
customestyle = 'mapbox://styles/bve81/ck8sikbwc27k61int0unl4xre'
token = "pk.eyJ1IjoiYnZlODEiLCJhIjoiY2s4c2QzeDJ6MGF4NzNlcGpmZ2pnajBpaSJ9.SqJSTzdrMoCl_upfZgC2cA"
# set the layout to plot

fig6 = px.scatter_mapbox(mapdata, lat = 'lat', lon = 'lon', size = 'cure',
                         color_discrete_sequence = ["fuchsia"], zoom = 3, height = 300)
fig6.update_layout(mapbox_style = customestyle, mapbox_accesstoken = token)
fig6.update_layout(margin = { "r": 0, "t": 0, "l": 0, "b": 0 })

# -----------------------------Tables
dt = d1 - dd
delta = [dd, d1, dt]

fig7 = go.Figure(
    data = [go.Table(header = dict(values = ['Статистистика Вчера', 'Статистистика Сегодня', 'Умерло за сутки']),
                     cells = dict(values = delta))
            ])
fig7.update_layout(title_text = 'Умерло за сутки')

# -----------------------------Dash parts
app = dash.Dash()
app.layout = html.Div([
    html.H1('Статистика по COVID19 Россия, данные из Johns Hopkins CSSE '),



    html.Div([
        html.H2(f"Количество заражений на {yesterday}", style={'text-align':'center'}),
        daq.LEDDisplay(
            id = 'digital',
            #label = f"Количество заражений на {yesterday}",
            style = { "fontSize": 40,'text-align':'center' },
            # labelPosition = 'bottom',
            value = d3,  # not sure what to put here
            size = 64,
            color = "#FF5E5E"
        )]),
    html.Div([
        dcc.Graph(id = 'Total case vs day', figure = fig9)]),

    html.Div([
        dcc.Graph(id = 'Total case dayli', figure = fig11)]),
    html.Div([
        dcc.Graph(id = 'Total death per day ', figure = figDYD)]),
    html.Div([
        dcc.Graph(id = 'Daily cases AI forecast', figure = figAI)]),
    html.Div([
        dcc.Graph(id = 'Daily cases AI forecast 02', figure = figAI02)]),
    html.Div([
        dcc.Graph(id = 'Active vs total', figure = fig10)]),
    html.Div([
        dcc.Graph(id = 'Total case vs day2', figure = figDD)]),
    html.Div([
        html.H2(f"Количество заражений за вычетом умерших и вылечившихся {yesterday}", style = { 'text-align': 'center' }),
        daq.LEDDisplay(
            id = 'digital2',
            # label = f"Количество заражений на {yesterday}",
            style = { "fontSize": 40, 'text-align': 'center' },
            # labelPosition = 'bottom',
            value = lastval,  # not sure what to put here
            size = 64,
            color = "#FF5E5E"
        )]),
    html.Div([
        dcc.Graph(id = 'delta for day ', figure = fig7)]),
    html.Div([
        dcc.Graph(id = 'Total death ', figure = fig3)]),

    html.Div([
        dcc.Graph(id = 'Total death Map', figure = fig6)]),

])

app.run_server(debug = True, use_reloader = False, host = '0.0.0.0', )  #
