import tweepy
import numpy as np
from keys import keys

import dataprep
from dataprep import *
import keras
from modelpredict import *

CONSUMER_KEY = keys['consumer_key']
CONSUMER_SECRET = keys['consumer_secret']
ACCESS_TOKEN = keys['access_token']
ACCESS_TOKEN_SECRET = keys['access_token_secret']

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

#api.update_status(' the dervish order dervishes is a disciple is sufism in the hand of the line. the sufi because of the consciousness in the ancient forms of the sufi sources of the sufi orders and his contemplation of the love ')

tweets = api.search_tweets('taylor swift') + api.search_tweets('Taylor swift') + api.search_tweets('Talyor Swift') + api.search_tweets('britain') + api.search_tweets('Britain') + api.search_tweets('cats')

cap = 1000

# extract each tweet and reply
for i, tweet in enumerate(tweets):
    input = clean_data_nasrudin(tweet.text)
    print(tweet)
    try: 
        user = tweet.entities['user_mentions'][0]['screen_name']

        # pick random number between 40 to 100 reponse characters
        text_len = np.random.randint(40, 100 + 1)

        # call model
        #response = generate_text(input, text_len)
        path = 'sufis_full.txt'
        index = 0
        maxChar = 40

        td = dataprep.TextData(path, index, maxChar)
        alphabet, char_to_int, int_to_char = td.get_parsed()

        # now call predict: input, model, text_len, maxChar, alphabet, char_to_int, int_to_char
        output = generate_text(input, text_len, maxChar, alphabet, char_to_int, int_to_char)

        # now post!
        #print('tweet', tweet)
        print('input', input)
        print('user', user)
        message = '@'+ user+output
        print('message', message)
        api.update_status(status = message, in_reply_to_status_id = tweet.id)

    except:
        print('something went wrong')
    
    if i > cap:
        break
