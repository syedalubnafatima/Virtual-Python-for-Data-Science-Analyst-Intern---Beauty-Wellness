import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('sentiment_results.csv')
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()
positive_text = ' '.join(df[df['sentiment'] == 'Positive']['processed_text'].dropna())
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positive Reviews')
plt.tight_layout()
plt.savefig('wordcloud_positive.png')
plt.show()


negative_text = ' '.join(df[df['sentiment'] == 'Negative']['processed_text'].dropna())

wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Negative Reviews')
plt.tight_layout()
plt.savefig('wordcloud_negative.png')
plt.show()

print("Visualizations saved: sentiment_distribution.png, wordcloud_positive.png, wordcloud_negative.png")