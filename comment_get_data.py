import requests
from bs4 import BeautifulSoup
import re

def get_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = soup.find_all('div', class_='content-txt review-card-content')
    return [review.get_text(strip=True) for review in reviews]

# İki farklı sayfadan yorumları çek
url = 'https://www.beyazperde.com/filmler/film-306940/kullanici-elestirileri/'
url2 = 'https://www.beyazperde.com/filmler/film-306940/kullanici-elestirileri/?page=2'

reviews_page1 = get_reviews(url)
reviews_page2 = get_reviews(url2)

# İki sayfanın yorumlarını birleştir
all_reviews = reviews_page1 + reviews_page2

# Yorumları bir dosyaya kaydet
with open('ExtractedData/comments_contents.txt', 'w', encoding='utf-8') as file:
    for review in all_reviews:
        file.write(review + '\n')