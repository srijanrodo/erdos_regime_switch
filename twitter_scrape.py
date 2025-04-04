import requests
import pandas as pd
from urllib.parse import quote
from datetime import datetime, timedelta

def scrape_tweets_requests(keywords, start_date, end_date, max_tweets=100):
    """
    Scrape tweets using requests library
    
    Args:
    - keywords (list): Search terms
    - start_date (str): Start date in YYYY-MM-DD format
    - end_date (str): End date in YYYY-MM-DD format
    - max_tweets (int): Maximum number of tweets to collect
    
    Returns:
    - pandas DataFrame with tweets
    """
    # Construct search query
    search_query = ' '.join([f'"{keyword}"' for keyword in keywords])
    
    # URL encode the search query
    encoded_query = quote(f'{search_query} since:{start_date} until:{end_date}')
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://twitter.com/',
        'DNT': '1'
    }
    
    # URLs to try
    urls = [
        f'https://twitter.com/search?q={encoded_query}&src=typed_query&f=live',
        f'https://x.com/search?q={encoded_query}&src=typed_query&f=live'
    ]
    
    tweets_list = []
    
    for url in urls:
        try:
            # Send GET request
            response = requests.get(url, headers=headers)
            
            # Check if request was successful
            if response.status_code == 200:
                # Save raw response for debugging
                with open('twitter_response.html', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                print(f"Response received. Status code: {response.status_code}")
                print(f"Response length: {len(response.text)} characters")
                
                # Basic text extraction (very basic and likely incomplete)
                import re
                
                # Try to extract potential tweet-like text
                tweet_patterns = [
                    r'"text":"(.*?)"',  # JSON-like text extraction
                    r'class="[^"]*tweet[^"]*">(.*?)<',  # HTML class-based extraction
                    r'<p[^>]*>(.*?)</p>'  # Paragraph text extraction
                ]
                
                found_tweets = []
                for pattern in tweet_patterns:
                    found_tweets.extend(re.findall(pattern, response.text, re.DOTALL))
                
                # Clean and filter tweets
                cleaned_tweets = []
                for tweet in found_tweets:
                    # Basic cleaning
                    tweet = re.sub(r'\\n', ' ', tweet)
                    tweet = re.sub(r'\\t', ' ', tweet)
                    tweet = re.sub(r'\\u[\da-fA-F]{4}', '', tweet)
                    
                    # Filter out very short or irrelevant tweets
                    if len(tweet) > 20 and 'Sign in' not in tweet:
                        cleaned_tweets.append({
                            'text': tweet,
                            'date': datetime.now().strftime('%Y-%m-%d')
                        })
                
                # Limit to max tweets
                tweets_list.extend(cleaned_tweets[:max_tweets])
                
                if tweets_list:
                    break
            else:
                print(f"Failed to retrieve page. Status code: {response.status_code}")
        
        except Exception as e:
            print(f"An error occurred: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets_list)
    return df

# Parameters
keywords = ["bitcoin", "btc"]
start_date = "2023-01-01"
end_date = "2023-01-31"

# Scrape tweets
try:
    tweets_df = scrape_tweets_requests(keywords, start_date, end_date, max_tweets=50)
    
    # Save to CSV
    tweets_df.to_csv('twitter_scrape_results.csv', index=False)
    
    # Display results
    print(f"Collected {len(tweets_df)} tweets")
    print("\nSample tweets:")
    print(tweets_df.head())

except Exception as e:
    print(f"An error occurred: {e}")