import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import os
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import scipy
import plotly.figure_factory as ff
import plotly.io as pio
import statsmodels
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


pio.templates.default = "plotly_dark"
mapbox_access_token = 'pk.eyJ1IjoiYnZlODEiLCJhIjoiY2s4c2QzeDJ6MGF4NzNlcGpmZ2pnajBpaSJ9.SqJSTzdrMoCl_upfZgC2cA'

#----------------------------------data load from source to pandas 
df=pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df.columns =[column.replace("/", "_") for column in df.columns]

deathdata=pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
deathdata.columns=[column.replace("/", "_") for column in deathdata.columns]

recoverydata=pd.read_csv(r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
recoverydata.columns=[column.replace("/", "_") for column in recoverydata.columns]

mapdata=pd.read_csv('c:\\city.csv')
currentdate = '9_27_20'
old=[ 5957.43108261, 6008.36468315, 6248.52659261, 6690.37627876, 7334.75013578]
#-----------------------------------focus on Russia
df=df.query('Country_Region == "Russia"')
deathdata=deathdata.query('Country_Region == "Russia"')
recoverydata=recoverydata.query('Country_Region == "Russia"')
#-----------------------------------Pie charts
fig2 = px.pie(df, values=currentdate, names='Country_Region', title='Количество заражений на 9/28/20', template="plotly_dark")
fig2.update_traces(textposition='inside', textinfo='value+label')

labels=['Умерло','Вылечилось'] #labels for pie chart
#labelst=['Умерло','Вылечилось', 'Заражений']#labels for pie chart
labelst=['Активных']
#-----------------------------------Value from tables

for value in deathdata['9_27_20']:
    dd=value
for value in deathdata[currentdate]:
    d1=value
for value in recoverydata[currentdate]:
    d2=value
for value in df[currentdate]:
    d3=value
recoverydata2=recoverydata.loc[:,'1_22_20': currentdate]
for i in recoverydata2.columns:
    for value in recoverydata2.values:
        rc=value
deathdata2=deathdata.loc[:,'1_22_20': currentdate]
for i in deathdata2.columns:
    for value in deathdata2.values:
        dt=value
    
for i in df.columns:
    for value in df.values:
        d=value
active= d[4:]-rc-dt

df2=df.loc[:,'1_22_20': currentdate].diff(axis=1)
for i in df2.columns:
    for value in df2.values:
        cc=value

#------------------------------------AI part forecasting
daily_cases = cc[1:]
#print(daily_cases)
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(daily_cases, axis=1))

all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)
seq_length = 5
X_all, y_all = create_sequences(all_data, seq_length)

X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()
class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred

def train_model(
  model,
  train_data,
  train_labels,
  test_data=None,
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 160

  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred = model(X_all)

    loss = loss_fn(y_pred.float(), y_all)

    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.float(), y_test)
      test_hist[t] = test_loss.item()

      if t % 10 == 0:
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 10 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

  return model.eval(), train_hist, test_hist



model = CoronaVirusPredictor(
  n_features=1,
  n_hidden=512,
  seq_len=seq_length,
  n_layers=2
)
model, train_hist, _ = train_model(model, X_all, y_all)


DAYS_TO_PREDICT = 5

with torch.no_grad():
  test_seq = X_all[:1]
  preds = []
  for _ in range(DAYS_TO_PREDICT):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()


print (predicted_cases)
#________________________________________________________________
recvsdead=[d1,d2]# Values for Pie charts
overall=[d3 - d2 - d1]# Values for Pie charts
print(overall)
#-----------------------Pie charts
fig3 = go.Figure(data=[go.Pie(labels=labels, values=recvsdead)])
fig3.update_traces(textposition='inside', textinfo='value+label')
fig3.update_layout(title_text="Количество умерших vs вылечившихся")

fig5 = go.Figure(data=[go.Pie(labels=labelst, values=overall)])
fig5.update_traces(textposition='inside', textinfo='value+label')
fig5.update_layout(title_text="Количество заражений - умерших/вылечившихся")

datelist = pd.date_range(start='1/22/2020', end='09/27/2020', tz=None).tolist() # List of dates
aidatelist = pd.date_range(start='09/16/2020', end='09/27/2020', tz=None).tolist()
oldlist= pd.date_range(start='04/23/2020', end='04/28/2020', tz=None).tolist()



    
#-------------------------Bar charts
#old=[ 3867.88543272,  3251.87410569,  2994.00512791,  4829.76698065,
 # 7897.2707901,  10192.6125507]

#print (df2)
fig9=make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)


fig9.add_trace(go.Bar(name='Total infected', x=datelist, y=d[4:]), 1, 1)
fig9.add_trace(go.Bar(name='Daily infected', x=datelist, y=cc), 1, 2)
# Change the bar mode
fig9.update_layout(barmode='stack')
fig9.update_layout(title_text='Количество инфицированных за день vs общее количество инфицированных')
fig9.add_trace(go.Scatter(x=datelist, y=d[4:],
                    mode='lines+markers',
                    name='Текущий тренд'),1,1)
fig9.add_trace(go.Scatter(x=datelist, y=cc,
                    mode='lines+markers',
                    name='Текущий тренд'),1,2)
figAI=make_subplots(rows=1, cols=1, specs=[[{}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

figAI.add_trace(go.Bar(name='Daily infected', x=datelist, y=cc),1,1)
figAI.add_trace(go.Scatter(x=datelist, y=cc,
                    mode='lines+markers',
                    name='Текущий тренд'),1,1)
figAI.add_trace(go.Scatter(x=aidatelist,  y=predicted_cases[0:5], #y=old, 
                    mode='lines+markers',
                    name='Будущий тренд'),1,1)
figAI.add_trace(go.Scatter(x=oldlist,  y=old[0:5], #y=old, 
                    mode='lines+markers',
                    name='Прошлый прогноз'),1,1)


fig10 = go.Figure(data=[
    go.Bar(name='Total infected', x=datelist, y=d[4:], width=[15], text=d[4:],textposition='auto'),
    go.Bar(name='Total infected - (cure/dead)', x=datelist, y=active, width=[15], text=active,textposition='auto')
])
# Change the bar mode
fig10.update_layout(barmode='group')
fig10.update_layout(title_text='Общее количество заражений в сравнении с активным колисеством случаев')
fig10.add_trace(go.Scatter(x=datelist, y=d[4:],
                    mode='lines+markers',
                    name='Текущий тренд'))
fig10.add_trace(go.Scatter(x=datelist, y=active,
                    mode='lines+markers',
                    name='Текущий тренд'))




fig = px.bar(df, x=datelist, y=d[4:], title='Рост количества заражений по датам на 09/15/20',
             labels={ # replaces default labels by column name
                "x": "Date",  "y": "Numder of Cases"
            }, template="plotly_dark")
fig.update_traces(showlegend=True,text=d[4:],textposition='auto', name='Кол-во заражнний')
fig.add_trace(go.Scatter(x=datelist, y=d[4:],
                    mode='lines+markers',
                    name='Текущий тренд'))


for i in deathdata.columns:
    for value in deathdata.values:
        d=value
fig4 = px.bar(deathdata, x=datelist, y=d[4:], title='Рост количества смертей по датам на 09/27/20',
             labels={ # replaces default labels by column name
                "x": "Date",  "y": "Numder of Cases"
            }, template="plotly_dark")

fig4.update_traces(text=d[4:],textposition='auto', showlegend=True, name='Кол-во Литальных случаев')
fig4.add_trace(go.Scatter(x=datelist, y=d[4:],
                    mode='lines+markers',
                    name='Тренд смертности'))

#-----------------------------Map charts
#set the geo=spatial data
customestyle='mapbox://styles/bve81/ck8sikbwc27k61int0unl4xre'
token="pk.eyJ1IjoiYnZlODEiLCJhIjoiY2s4c2QzeDJ6MGF4NzNlcGpmZ2pnajBpaSJ9.SqJSTzdrMoCl_upfZgC2cA"
#set the layout to plot

fig6 = px.scatter_mapbox(mapdata, lat='lat', lon='lon', size='cure',
                        color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig6.update_layout(mapbox_style=customestyle,  mapbox_accesstoken=token)
fig6.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

#-----------------------------Tables
dt=d1-dd
delta=[dd,d1,dt]

fig7 = go.Figure(data=[go.Table(header=dict(values=['Статистистика Вчера', 'Статистистика Сегодня', 'Умерло за сутки']),
                 cells=dict(values=delta))
                     ])
fig7.update_layout(title_text='Умерло за сутки')


#-----------------------------Dash parts
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
                 
  #  dash_table.DataTable(
   # id='table',
   # columns=[{"name": i, "id": i} for i in df.columns],
    #data=df.to_dict('records'))
                 ])




app.run_server(debug=True, use_reloader=False, host='0.0.0.0', )  # 
