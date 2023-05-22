import string
import nltk
import re
import pandas as pd

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download('punkt') -> only for the first use
# nltk.download('stopwords') -> only for the first use
# nltk.download('vader_lexicon') -> only for the first use


def main():
    analyze_tweets()


def analyze_tweets():
    stop_words = set(stopwords.words('english'))
    sentiment_analyzer = SentimentIntensityAnalyzer()
    df = pd.read_csv('tweets_after.csv')

    results_dict = defaultdict(lambda: defaultdict(int))
    for i, row in df.iterrows():
        name = row['handle']
        tweet = row['text']

        # manipulate the dataset
        tweet = remove_hashtag(tweet)
        tweet = remove_at(tweet)
        tweet = remove_url(tweet)
        tweet = remove_punctuation(tweet)
        tokens = tokenize(tweet)
        tokens = remove_stopwords(tokens, stop_words)

        # analyze the sentiment
        result = sentiment(' '.join(tokens), sentiment_analyzer)['compound']

        if result < -0.1:
            result = 'negative'
        elif result > 0.1:
            result = 'positive'
        else:
            result = 'neutral'

        results_dict[name][result] += 1

    df2 = pd.DataFrame([{'name': 'Donald Trump' if n == 'realDonaldTrump' else 'Hillary Clinton',
                         'sentiment': r, 'count': c}
                        for n, r in results_dict.items()
                        for r, c in r.items()])

    df2.to_csv('tweet_sentiments.csv', index=False)


def sentiment(text, sentiment_analyzer):
    return sentiment_analyzer.polarity_scores(text)


def remove_hashtag(text):
    return re.sub(r'#\S+', '', text)


def remove_at(text):
    return re.sub(r'@\S+', '', text)


def remove_url(text):
    return re.sub(r'http\S+|www.\S+', '', text)


def remove_stopwords(tokens, stop_words):
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


if __name__ == "__main__":
    main()
