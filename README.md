# covid19
COVID19 Russian analytics \
Analytics of COVID data from John Hopkins University data source only for Russia region. Python, AI forecasting ETNA (Tinkoff). <BR />
Web service running on http://0.0.0.0:8050 AI forecasting is for 8 days <BR />
For update data you need restart covid19.py every 24 hours <BR />
For docker container run this in cmd: 
```
docker pull bve81/covid19
docker run -p 8050:8050 bve81/covid19
```
Open http://0.0.0.0:8050 in web browser
