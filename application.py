import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used to analyze sentiments of tweets 🐦")
st.sidebar.markdown("This application is a Streamlit dashboard used to analyze sentiments of tweets 🐦")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv('Tweets.csv')
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

# Sidebar random tweet
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.write(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

# Visualization for tweet counts
st.sidebar.subheader("Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if select == 'Bar plot':
    fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
else:
    fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
st.plotly_chart(fig)

# Word Cloud
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
df = data[data['airline_sentiment'] == word_sentiment]
words = ' '.join(df['text'])
processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()
