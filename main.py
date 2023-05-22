import string
import nltk

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download('punkt') -> only for the first use
# nltk.download('stopwords') -> only for the first use
# nltk.download('vader_lexicon') -> only for the first use


def main():
    stop_words = set(stopwords.words('english'))
    sentiment_analyzer = SentimentIntensityAnalyzer()



def sentiment(text, sentiment_analyzer):
    return sentiment_analyzer.polarity_scores(text)


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
