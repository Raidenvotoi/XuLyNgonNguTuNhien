import asyncio
import asyncpraw

client_id = "cfgRNdcf6Vw7PF9mBKrWKA"
client_secret = "k8YD6GloeBp5uSvV0VHOMknpyjjIrA"
user_agent = "conheo"

async def main():
    reddit = asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    subreddit = await reddit.subreddit("Roadcam")
    
    async for submission in subreddit.hot(limit=10):  # Sử dụng async for
        print(submission.title)
        print()

        await submission.load()  # Tải dữ liệu bài đăng

        for comment in submission.comments:
            if isinstance(comment, asyncpraw.models.MoreComments):
                continue  # Bỏ qua các comment chưa tải hết
            print(comment.body)
        break  # Chỉ lấy 1 bài viết đầu tiên

    await reddit.close()  # Đóng kết nối

# Chạy hàm async bằng asyncio
asyncio.run(main())
