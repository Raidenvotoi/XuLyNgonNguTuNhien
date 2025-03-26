# Import necessary modules
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Initialize empty lists to store movie names and ratings
movie_name = []
Rating = []

# Set the URL to scrape
url = "https://www.imdb.com/search/title/?groups=top_100&sort=user_rating,desc"

# Send an HTTP request to the URL and store the response
response = requests.get(url)

# Parse the HTML content of the response
soup = BeautifulSoup(response.content, "html.parser")

# Find all the <div> elements with the class "lister-item mode-advanced"
movies = soup.find_all('div', class_='lister-item mode-advanced')

# For each movie, extract the title and rating
for movie in movies:
    title = movie.h3.a.text
    movie_name.append(title)
    
    rating = movie.strong.text if movie.strong else None
    Rating.append(rating)

# Show results
dic = {"Movie Name": movie_name, "Rating": Rating}
df = pd.DataFrame(dic)

# Display the top 25 movies
print(df.head(25))
