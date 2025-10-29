import os
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text

neo_uri = os.getenv("NEO4J_URI")
neo_user = os.getenv("NEO4J_USER")
neo_pass = os.getenv("NEO4J_PASSWORD")
db_url = os.getenv("DATABASE_URL")

print("Checking Neo4j...")
driver = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pass))
with driver.session() as session:
    result = session.run("RETURN 'Neo4j OK' AS status")
    print(result.single()["status"])

print("Checking PostgreSQL...")
engine = create_engine(db_url)
with engine.connect() as conn:
    result = conn.execute(text("SELECT 'Postgres OK'")).scalar()
    print(result)