# from db.postgres_handler import PostgresHandler
# from db.chroma_handler import ChromaHandler
from datetime import datetime, timedelta
# from connectors.event_registry_connector import EventRegistryConnector
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
from jobs_extraction import extract_jobs_from_table
from news_extractions import fetch_techcrunch_news
from tag_query_classification import classify_tags
from tag_query_classification import classify_query
import re
import os
import feedparser
import json
import requests
import trafilatura
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv(override=True)


def main():
     
    
    user_query = input("Enter your search query: ")
    
    query_type = classify_query(user_query)
    
    if query_type == "jobs":
        jobs = extract_jobs_from_table()
        for job in jobs[:20]:
            print("COMPANY:", job["company"])
            print("ROLE:", job["role"])
            print("LOCATION:", job["location"])
            print("APPLY LINK:", job["apply_link"])
            print("DATE POSTED:", job["date_posted"])
            print("-" * 80)
        
    else:
        
        result = classify_tags(user_query)
        
        tags = set(result["companies"] + result["topics"])
        
        all_news = []
        for tag in tags:
            all_news.extend(fetch_techcrunch_news(tag))

        for item in all_news:
            print(f"- {item['title']}: {item['description']}")  
    
    
if __name__ == "__main__":
    main()
