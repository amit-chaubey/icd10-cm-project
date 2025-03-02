import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from neo4j import GraphDatabase
import os
import sys
from dotenv import load_dotenv
import time

# Add the project root directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Now we can import from backend
from backend.db.neo4j_client import Neo4jClient

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load processed ICD data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clear_database(client: Neo4jClient) -> bool:
    """Clear all data from the database"""
    try:
        with client.driver.session() as session:
            # Drop existing constraints
            constraints = session.run("SHOW CONSTRAINTS").data()
            for constraint in constraints:
                constraint_name = constraint.get('name')
                if constraint_name:
                    session.run(f"DROP CONSTRAINT {constraint_name}")
                    logger.info(f"Dropped constraint: {constraint_name}")

            # Remove all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            
            # Verify deletion
            count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            if count == 0:
                logger.info("Database cleared successfully")
                return True
            else:
                logger.error(f"Database not fully cleared. {count} nodes remaining.")
                return False
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return False

def verify_load(client):
    """Verify data was loaded correctly"""
    with client.driver.session() as session:
        stats = session.run("""
            MATCH (v:Version) WITH count(v) as versions
            MATCH (l:Letter) WITH versions, count(l) as letters
            MATCH (t:Term) WITH versions, letters, count(t) as terms
            MATCH (c:Code) WITH versions, letters, terms, count(c) as codes
            RETURN versions, letters, terms, codes
        """).single()
        
        logger.info("\nDatabase Statistics:")
        logger.info(f"Versions: {stats['versions']}")
        logger.info(f"Letters: {stats['letters']}")
        logger.info(f"Terms: {stats['terms']}")
        logger.info(f"Codes: {stats['codes']}")

def load_data():
    """Load ICD-10 data into Neo4j"""
    try:
        # Initialize client
        client = Neo4jClient()
        
        # Load JSON data
        data_path = Path(__file__).parent.parent.parent / "data/processed/indexed/icd10cm_index.json"
        with open(data_path) as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data structure")
            
        with client.driver.session() as session:
            # Clear existing data
            response = input("Clear existing data? (yes/no): ").lower().strip()
            if response == 'yes':
                if client.clear_database():
                    logger.info("Database cleared successfully")
                else:
                    logger.error("Failed to clear database")
                    return
            
            # Create constraints
            client.create_constraints()
            
            # Create Version node
            session.run("""
                CREATE (v:Version {number: $version, title: $title})
            """, version=data['version'], title=data['title'])
            
            # Process letters and terms
            for letter_name, letter_data in data['terms'].items():
                logger.info(f"Processing letter: {letter_name}")
                
                # Create letter node
                session.run("""
                    MERGE (l:Letter {name: $letter})
                """, letter=letter_name)
                
                # Process terms
                for term in letter_data['terms']:
                    client.create_term_hierarchy(session, term, letter_name)
            
            logger.info("Data loading complete")
            verify_load(client)
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    load_data()