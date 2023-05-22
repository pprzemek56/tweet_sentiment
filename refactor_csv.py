import pandas as pd

df = pd.read_csv('tweets_before.csv')

df = df.drop(['id', 'is_retweet', 'original_author', 'time', 'in_reply_to_screen_name',
              'in_reply_to_status_id', 'in_reply_to_user_id', 'is_quote_status', 'lang',
              'retweet_count', 'favorite_count', 'longitude', 'latitude', 'place_id',
              'place_full_name', 'place_name', 'place_type', 'place_country_code', 'place_country',
              'place_contained_within', 'place_attributes', 'place_bounding_box', 'source_url',
              'truncated', 'entities', 'extended_entities'], axis=1)

df.to_csv('tweets_after.csv', index=False)
