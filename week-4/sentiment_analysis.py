import pandas as pd
from textblob import TextBlob

df = pd.read_csv('preprocessed_reviews.csv')

def get_sentiment(text):
    if pd.isna(text) or text.strip() == '':
        return 'Neutral'
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['processed_text'].apply(get_sentiment)
print(df['sentiment'].value_counts())
print(df[['processed_text', 'sentiment']].head(10))

df.to_csv('sentiment_results.csv', index=False)
print("Sentiment analysis complete. Saved to sentiment_results.csv")