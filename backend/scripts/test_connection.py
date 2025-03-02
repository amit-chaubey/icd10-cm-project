from backend.db.neo4j_client import Neo4jClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    try:
        client = Neo4jClient()
        with client.driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message")
            print(result.single()['message'])
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    test_connection()