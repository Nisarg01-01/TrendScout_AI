import time
import schedule
import subprocess
import config
from datetime import datetime

def run_ingestion():
    print(f"[{datetime.now()}] Starting ingestion cycle...")
    
    try:
        print("Running ingest_news.py...")
        subprocess.run(["python", "ingest_news.py"], check=True)
        
        print("Running ingest_jobs.py...")
        subprocess.run(["python", "ingest_jobs.py"], check=True)
        
        print(f"[{datetime.now()}] Ingestion cycle completed successfully.")
        
        # In the future, trigger preprocess -> extract -> graph here
        
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Error during ingestion: {e}")

def main():
    print(f"Scheduler started. Running every {config.INGEST_INTERVAL_HOURS} hours.")
    
    # Run immediately on startup
    run_ingestion()
    
    # Schedule periodic runs
    schedule.every(config.INGEST_INTERVAL_HOURS).hours.do(run_ingestion)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
