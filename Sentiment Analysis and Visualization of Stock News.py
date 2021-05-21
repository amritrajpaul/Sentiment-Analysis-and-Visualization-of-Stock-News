from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ['HDB','IBN']
news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url = url,headers={'user-agent':'my-proj'})
    response = urlopen(req)

    html = BeautifulSoup(response,'html',features="html.parser")
    news_table = html.find(id="news-table")
    news_tables[ticker] = news_table
    
parsed_data = []

for ticker, news_table in news_tables.items(): #.items as it has to go through each key value pair
    for row in news_table.findAll('tr'):
        title = row.a.get_text()
        date_data = row.td.text.split(" ")
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        parsed_data.append([ticker,date,time,title])

df = pd.DataFrame(parsed_data,columns=['ticker','date','time','title'])
vader = SentimentIntensityAnalyzer()
f = lambda x:vader.polarity_scores(x)['compound']
df['compound'] = df['title'].apply(f)
df['date'] = pd.to_datetime(df.date).dt.date

mean_df = df.groupby(['ticker','date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound',axis='columns').transpose()
mean_df.plot(kind = 'bar')
plt.show()