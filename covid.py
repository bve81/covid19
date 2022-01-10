import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc
from dash import html
from plotly.subplots import make_subplots
from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.pipeline import Pipeline

pio.templates.default = "plotly_dark"
mapbox_access_token = 'pk.eyJ1IjoiYnZlODEiLCJhIjoiY2s4c2QzeDJ6MGF4NzNlcGpmZ2pnajBpaSJ9.SqJSTzdrMoCl_upfZgC2cA'

# ----------------------------------data load from source to pandas
df = pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                 r'/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df.columns = [column.replace("/", "_") for column in df.columns]

deathdata = pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                        r'/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
deathdata.columns = [column.replace("/", "_") for column in deathdata.columns]

recoverydata = pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                           r'/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
recoverydata.columns = [column.replace("/", "_") for column in recoverydata.columns]

mapdata = pd.read_csv('city.csv')
currentdate = '1_9_22'
old = [5957.43108261, 6008.36468315, 6248.52659261, 6690.37627876, 7334.75013578]
# -----------------------------------focus on Russia
df = df.query('Country_Region == "Russia"')
deathdata = deathdata.query('Country_Region == "Russia"')
recoverydata = recoverydata.query('Country_Region == "Russia"')
# -----------------------------------Pie charts
fig2 = px.pie(df, values=currentdate, names='Country_Region', title='Количество заражений на 01/09/22',
              template="plotly_dark")
fig2.update_traces(textposition='inside', textinfo='value+label')

labels = ['Умерло', 'Вылечилось']  # labels for pie chart

labelst = ['Активных']
datelist = pd.date_range(start='1/22/2020', end='1/9/2022', tz=None).tolist()  # List of dates
# -----------------------------------Value from tables

for value in deathdata['1_8_22']:
    dd = value
for value in deathdata[currentdate]:
    d1 = value
for value in recoverydata[currentdate]:
    d2 = value
for value in df[currentdate]:
    d3 = value
recoverydata2 = recoverydata.loc[:, '1_22_20': currentdate]
for i in recoverydata2.columns:
    for value in recoverydata2.values:
        rc = value
deathdata2 = deathdata.loc[:, '1_22_20': currentdate]
for i in deathdata2.columns:
    for value in deathdata2.values:
        dt = value

for i in df.columns:
    for value in df.values:
        d = value
active = d[4:] - rc - dt

df2 = df.loc[:, '1_22_20': currentdate].diff(axis=1)
for i in df2.columns:
    for value in df2.values:
        cc = value

# AI etna Forecast
# Read the data
dataAI = {'dates': datelist, 'values': cc}
original_df = pd.DataFrame(data=dataAI)
original_df[ "timestamp" ] = pd.to_datetime(original_df[ "dates" ])
print(original_df[ "timestamp" ])
original_df[ "target" ] = original_df[ "values" ]
original_df.drop(columns=[ "dates", "values" ], inplace=True)
original_df[ "segment" ] = "main"
original_df.head()


# Create a TSDataset
df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="D")

# Choose a horizon
HORIZON = 8

# Fit the pipeline
pipeline = Pipeline(model=ProphetModel(yearly_seasonality=True, daily_seasonality=True), horizon=HORIZON)
pipeline.fit(ts)

# Make the forecast
forecast_ts = pipeline.forecast()
dfs = forecast_ts.to_pandas(flatten=True)
print(dfs)

# ________________________________________________________________
recvsdead = [d1, d2]  # Values for Pie charts
overall = [d3 - d2 - d1]  # Values for Pie charts
print(overall)
# -----------------------Pie charts
fig3 = go.Figure(data=[go.Pie(labels=labels, values=recvsdead)])
fig3.update_traces(textposition='inside', textinfo='value+label')
fig3.update_layout(title_text="Количество умерших vs вылечившихся")

fig5 = go.Figure(data=[go.Pie(labels=labelst, values=overall)])
fig5.update_traces(textposition='inside', textinfo='value+label')
fig5.update_layout(title_text="Количество заражений - умерших/вылечившихся")

aidatelist = pd.date_range(start='01/10/2022', end='01/17/2022', tz=None).tolist()
oldlist = pd.date_range(start='01/09/2022', end='01/16/2022', tz=None).tolist()

# -------------------------Bar charts


fig9 = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                     shared_yaxes=False, vertical_spacing=0.001)

fig9.add_trace(go.Bar(name='Total infected', x=datelist, y=d[4:]), 1, 1)
fig9.add_trace(go.Bar(name='Daily infected', x=datelist, y=cc), 1, 2)
# Change the bar mode
fig9.update_layout(barmode='stack')
fig9.update_layout(title_text='Количество инфицированных за день vs общее количество инфицированных')
fig9.add_trace(go.Scatter(x=datelist, y=d[4:],
                          mode='lines+markers',
                          name='Текущий тренд'), 1, 1)
fig9.add_trace(go.Scatter(x=datelist, y=cc,
                          mode='lines+markers',
                          name='Текущий тренд'), 1, 2)
figAI = make_subplots(rows=1, cols=1, specs=[[{}]], shared_xaxes=True,
                      shared_yaxes=False, vertical_spacing=0.001)
#
figAI.add_trace(go.Bar(name='Daily infected', x=datelist, y=cc), 1, 1)
figAI.add_trace(go.Scatter(x=datelist, y=cc,
                           mode='lines+markers',
                           name='Текущий тренд'), 1, 1)
figAI.add_trace(go.Scatter(x=aidatelist, y=dfs["target"],  # y=old,
                           mode='lines+markers',
                           name='Будущий тренд'), 1, 1)
# figAI.add_trace(go.Scatter(x=oldlist,  y=old[0:5], #y=old,
#                     mode='lines+markers',
#                     name='Прошлый прогноз'),1,1)


fig10 = go.Figure(data=[
    go.Bar(name='Total infected', x=datelist, y=d[4:], width=[15], text=d[4:], textposition='auto'),
    #  go.Bar(name='Total infected - (cure/dead)', x=datelist, y=active, width=[15], text=active, textposition='auto')
])
# Change the bar mode
fig10.update_layout(barmode='group')
fig10.update_layout(title_text='Общее количество заражений в сравнении с активным колисеством случаев')
fig10.add_trace(go.Scatter(x=datelist, y=d[4:],
                           mode='lines+markers',
                           name='Текущий тренд'))
# fig10.add_trace(go.Scatter(x=datelist, y=active,
#                            mode='lines+markers',
#                            name='Текущий тренд'))

fig = px.bar(df, x=datelist, y=d[4:], title='Рост количества заражений по датам на 01/09/22',
             labels={  # replaces default labels by column name
                 "x": "Date", "y": "Numder of Cases"
             }, template="plotly_dark")
fig.update_traces(showlegend=True, text=d[4:], textposition='auto', name='Кол-во заражнний')
fig.add_trace(go.Scatter(x=datelist, y=d[4:],
                         mode='lines+markers',
                         name='Текущий тренд'))

for i in deathdata.columns:
    for value in deathdata.values:
        d = value
fig4 = px.bar(deathdata, x=datelist, y=d[4:], title='Рост количества смертей по датам на 01/09/22',
              labels={  # replaces default labels by column name
                  "x": "Date", "y": "Numder of Cases"
              }, template="plotly_dark")

fig4.update_traces(text=d[4:], textposition='auto', showlegend=True, name='Кол-во Литальных случаев')
fig4.add_trace(go.Scatter(x=datelist, y=d[4:],
                          mode='lines+markers',
                          name='Тренд смертности'))

# -----------------------------Map charts
# set the geo=spatial data
customestyle = 'mapbox://styles/bve81/ck8sikbwc27k61int0unl4xre'
token = "pk.eyJ1IjoiYnZlODEiLCJhIjoiY2s4c2QzeDJ6MGF4NzNlcGpmZ2pnajBpaSJ9.SqJSTzdrMoCl_upfZgC2cA"
# set the layout to plot

fig6 = px.scatter_mapbox(mapdata, lat='lat', lon='lon', size='cure',
                         color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig6.update_layout(mapbox_style=customestyle, mapbox_accesstoken=token)
fig6.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

# -----------------------------Tables
dt = d1 - dd
delta = [dd, d1, dt]

fig7 = go.Figure(data=[go.Table(header=dict(values=['Статистистика Вчера', 'Статистистика Сегодня', 'Умерло за сутки']),
                                cells=dict(values=delta))
                       ])
fig7.update_layout(title_text='Умерло за сутки')

# -----------------------------Dash parts
app = dash.Dash()
app.layout = html.Div([
    html.H1('Статистика по COVID19 Россия, данные из Johns Hopkins CSSE '),
    html.Div([
        dcc.Graph(id='cases', figure=fig2)]),
    html.Div([
        dcc.Graph(id='Total case vs day', figure=fig9)]),
    html.Div([
        dcc.Graph(id='Daily cases AI forecast', figure=figAI)]),
    html.Div([
        dcc.Graph(id='Active vs total', figure=fig10)]),
    html.Div([
        dcc.Graph(id='cases by date ', figure=fig)]),

    html.Div([
        dcc.Graph(id='Total death vs cure vs total ', figure=fig5)]),
    html.Div([
        dcc.Graph(id='delta for day ', figure=fig7)]),
    html.Div([
        dcc.Graph(id='Total death ', figure=fig3)]),
    html.Div([
        dcc.Graph(id='Total death trend', figure=fig4)]),

    html.Div([
        dcc.Graph(id='Total death Map', figure=fig6)]),

])

app.run_server(debug=True, use_reloader=False, host='0.0.0.0', )  #
