import praw

client_id = "cfgRNdcf6Vw7PF9mBKrWKA"
client_secret = "k8YD6GloeBp5uSvV0VHOMknpyjjIrA"
user_agent = "conheo"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

subreddit = reddit.subreddit("Advice")
u=0
for submission in subreddit.hot(limit=10):  # Sử dụng vòng lặp đồng bộ
    print(submission.title)
    print(submission.selftext)

    submission.comments.replace_more(limit=0)  # Tải toàn bộ comment
    
    for comment in submission.comments.list():  # Lấy danh sách comment
        print(f"cmt {u}",comment.body)
        u+=1
    break  # Chỉ lấy 1 bài viết đầu tiên
# Đóng kết nối