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

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('Tweets.csv')
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    data['hour'] = data['tweet_created'].dt.hour
    data['day'] = data['tweet_created'].dt.day_name()
    data['tweet_length'] = data['text'].str.len()
    return data

data = load_data()

# Sidebar random tweet
st.sidebar.subheader("Show Random Tweet")
st.markdown("### Random Tweet by Sentiment")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.write(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

# Data Table
if st.sidebar.checkbox("Show Raw Data", False):
    st.markdown("### Raw Data")
    st.write(data)

# Visualization for tweet counts
st.sidebar.subheader("Number of Tweets by Sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
st.markdown("### Number of Tweets by Sentiment")

sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if select == 'Bar plot':
    fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
else:
    fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
st.plotly_chart(fig)

# Sentiment over time
st.sidebar.subheader("Sentiment Over Time")
if st.sidebar.checkbox("Show Sentiment Over Time"):
    st.markdown("### Sentiment Over Time")
    sentiment_over_time = data.groupby(data['tweet_created'].dt.date)['airline_sentiment'].value_counts().unstack()
    st.line_chart(sentiment_over_time)


# Word Cloud
st.sidebar.header("Word Cloud")
st.markdown("### Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
df = data[data['airline_sentiment'] == word_sentiment]
words = ' '.join(df['text'])
processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)

fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Geographic Analysis
if 'latitude' in data.columns:
    st.sidebar.subheader("Geographic Analysis")
    if st.sidebar.checkbox("Show Tweet Locations"):
        st.markdown("### Geographic Analysis of Tweets")
        st.map(data[['latitude', 'longitude']].dropnaa())

# Sentiment by airline
st.sidebar.subheader("Compare Airlines")
st.markdown("### Sentiment Comparison by Airline")
airlines = st.sidebar.multiselect("Select Airlines", data['airline'].unique(), default=data['airline'].unique())
if airlines:
    filtered_data = data[data['airline'].isin(airlines)]
    sentiment_by_airline = filtered_data.groupby(['airline', 'airline_sentiment']).size().unstack()
    st.bar_chart(sentiment_by_airline)


# Bubble Chart for Sentiments
st.sidebar.subheader("Advanced Visualizations")
if st.sidebar.checkbox("Show Bubble Chart"):
    st.markdown("### Bubble Chart of Tweets")
    fig = px.scatter(
        data, 
        x='tweet_length', 
        y='retweet_count',  # Use the correct column name here
        size='airline_sentiment_confidence',  # Adjust this to a relevant column, if 'likes' doesn't exist
        hover_name='text', 
        title="Bubble Chart of Tweets"
    )
    st.plotly_chart(fig)

# Heatmap for tweet frequencies
if st.sidebar.checkbox("Show Heatmap"):
    st.markdown("### Heatmap of Tweet Frequencies")
    heatmap_data = data.groupby(['day', 'hour']).size().unstack()
    fig = px.imshow(heatmap_data, labels=dict(x="Hour", y="Day", color="Tweet Count"))
    st.plotly_chart(fig)


# Export Filtered Data
st.sidebar.subheader("Export Filtered Data")
if st.sidebar.checkbox("Download Filtered Data"):
    st.markdown("### Download Filtered Data")
    filtered_csv = data.to_csv(index=False)
    st.download_button("Download CSV", filtered_csv, file_name="filtered_tweets.csv", mime="text/csv")

# # Sentiment Prediction (Mock Example)
# st.sidebar.subheader("Sentiment Prediction")
# user_tweet = st.text_input("Enter a Tweet to Predict Sentiment:")
# if user_tweet:
#     st.write("Predicted Sentiment: Positive (Mock Prediction)")

st.sidebar.markdown("-----")
st.sidebar.write("Created with ❤️ using Streamlit")
