import requests
import re
from bs4 import BeautifulSoup

def fetch_techcrunch_news(tag):
    url = f"https://techcrunch.com/tag/{tag}/feed/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")

    news_items = []

    for item in soup.find_all("item"):
        title = item.title.text if item.title else ""
        description = item.description.text if item.description else ""

        news_items.append({
            "title": title,
            "description": description
        })

    return news_items