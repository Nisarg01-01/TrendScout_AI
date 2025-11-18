import chromadb
from chromadb.config import Settings
from news_extractions import fetch_techcrunch_news

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="chroma_storage" 
    )
)

news_collection = client.get_or_create_collection(
    name="techcrunch_news"
)



def store_news_in_chroma(news_items):
    ids = []
    documents = breeze = []
    metadatas = []

    for idx, item in enumerate(news_items):
        unique_id = f"news-{item['title'][:40].replace(' ', '_')}-{idx}"

        ids.append(unique_id)
        documents.append(f"{item['title']} - {item['description']}")
        metadatas.append({
            "title": item["title"],
            "description": item["description"]
        })

    news_collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    