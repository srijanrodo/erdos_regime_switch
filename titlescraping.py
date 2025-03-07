import praw
import datetime
import csv
import os

start_date = datetime.datetime(2020, 1, 1)
end_date   = datetime.datetime(2025, 1, 31)


start_ts = start_date.timestamp()
end_ts   = end_date.timestamp()

reddit = praw.Reddit(
    client_id="a6ukBdG7-cCyxqRSfmO3VQ",
    client_secret="COKGjQE8a_zt3sRicfSwafXmYGm8aQ",
    password="springerdos2025",
    user_agent="testscript by u/fakebot3",
    username="moh_maya_sentiment",
)

subreddit = reddit.subreddit('cryptocurrencytrading')

desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
output_file = os.path.join(desktop_path, 'posts_output.csv')

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['title', 'content', 'upvotes', 'date_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Iterate over posts in descending order (newest first)
    for submission in subreddit.new(limit=None):
        # Because posts are returned newest first, stop once we're past the start date
        if submission.created_utc < start_ts:
            break

        # Process only posts within the specified time range
        if start_ts <= submission.created_utc <= end_ts:
            title = submission.title
            content = submission.selftext  # Note: may be empty for link posts
            upvotes = submission.score
            post_date = datetime.datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

            
            writer.writerow({
                'title': title,
                'content': content,
                'upvotes': upvotes,
                'date_time': post_date
            })

safe_print(f"CSV export completed. File saved at: {output_file}")
