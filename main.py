import string
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


# nltk.download('punkt') -> only for the first use
# nltk.download('stopwords') -> only for the first use
# nltk.download('vader_lexicon') -> only for the first use


def main():
    graph_visualization()


def graph_visualization():
    df = pd.read_csv('tweet_sentiments.csv')
    clinton_df = df[df['name'] == 'Hillary Clinton']
    trump_df = df[df['name'] == 'Donald Trump']
    clinton_dict = clinton_df.set_index('sentiment')['count'].to_dict()
    trump_dict = trump_df.set_index('sentiment')['count'].to_dict()

    groups = list(clinton_dict.keys())

    clinton_values = list(clinton_dict.values())
    trump_values = list(trump_dict.values())

    num_groups = len(groups)

    bar_width = 0.4

    fig, ax = plt.subplots()

    clinton_positions = [i for i in range(num_groups)]
    trump_positions = [i + bar_width for i in range(num_groups)]

    clinton_bars = ax.bar(clinton_positions, clinton_values, bar_width, label='Clinton')
    trump_bars = ax.bar(trump_positions, trump_values, bar_width, label='Trump')

    def add_counts(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    # Annotate counts below bars
    add_counts(clinton_bars)
    add_counts(trump_bars)

    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title("Sentiment analysis of Donald Trump's and Hillary Clinton's tweets")
    ax.set_xticks([r + bar_width / 2 for r in range(num_groups)])
    ax.set_xticklabels(groups)
    ax.legend()

    plt.show()


def analyze_tweets():
    stop_words = set(stopwords.words('english'))
    sentiment_analyzer = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
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
        tokens = lemmatize(tokens, lemmatizer)

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


def lemmatize(tokens, lemmatizer):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


if __name__ == "__main__":
    main()
