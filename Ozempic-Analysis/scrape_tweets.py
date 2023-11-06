import tweepy
from dotenv import load_dotenv
import os

load_dotenv()

# Retrieve API Keys
consumer_key = os.getenv('API_KEY')
consumer_secret = os.getenv('API_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_SECRET_TOKEN')

# Authenticate with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create Tweepy API Object
api = tweepy.API(auth)

# Define the search queries per the imported query list
# All relative search query topics are stored in the local dictionary file
search_queries = ['query1', 'query2', 'query3']  # Add your desired queries

# Fetch tweets for each search query
for search_query in search_queries:
    tweets = api.search(q=search_query, count=100)

    print(f"Tweets for '{search_query}':")
    for tweet in tweets:
        print(f"{tweet.user.screen_name}: {tweet.text}")
    print("\n")

# Close the connection
api = None
