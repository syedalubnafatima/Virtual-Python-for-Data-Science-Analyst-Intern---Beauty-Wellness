import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('nyka_top_brands_cosmetics_product_reviews.csv')
df = df[['review_text', 'review_rating', 'brand_name', 'product_title']]
df = df.dropna(subset=['review_text'])
df = df.sample(n=3000, random_state=42).reset_index(drop=True)
print(df.shape)
print(df.head())
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['review_text'].apply(clean_text)
print(df[['review_text', 'cleaned_text']].head())

df['tokens'] = df['cleaned_text'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]
df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
df['lemmatized_tokens'] = df['tokens_no_stopwords'].apply(lemmatize_tokens)
df['processed_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
print(df[['cleaned_text', 'tokens', 'tokens_no_stopwords', 'lemmatized_tokens', 'processed_text']].head())
df.to_csv('preprocessed_reviews.csv', index=False)
print("Preprocessing complete. Saved to preprocessed_reviews.csv")