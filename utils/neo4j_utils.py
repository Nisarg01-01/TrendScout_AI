import os
from neo4j import GraphDatabase
import config

def get_neo4j_driver():
    """Establishes connection to the Neo4j database."""
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )

def clear_database(driver):
    """Deletes all nodes and relationships in the database."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Cleared the Neo4j database.")

if __name__ == '__main__':
    # For direct testing
    driver = get_neo4j_driver()
    clear_database(driver)
    driver.close()
    print("Database cleared successfully.")
