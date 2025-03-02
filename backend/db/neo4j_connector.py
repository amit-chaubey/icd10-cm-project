"""
Neo4j database connector for the ICD-10-CM coding system.
"""
import logging
from typing import Dict, List, Optional, Union

import yaml
from neo4j import GraphDatabase, Session

logger = logging.getLogger(__name__)

class Neo4jConnector:
    """Handles connections and queries to the Neo4j graph database."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the Neo4j connector with configuration."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                
            neo4j_config = config.get("neo4j", {})
            self.uri = neo4j_config.get("uri", "bolt://localhost:7687")
            self.user = neo4j_config.get("user", "neo4j")
            self.password = neo4j_config.get("password", "password")
            
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, "driver"):
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def verify_connectivity(self) -> bool:
        """Verify that the database connection is working."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Database connectivity test failed: {str(e)}")
            return False
    
    def setup_schema(self):
        """Set up the database schema with constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT diagnosis_name IF NOT EXISTS FOR (d:Diagnosis) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT code_value IF NOT EXISTS FOR (c:Code) REQUIRE c.code IS UNIQUE",
            "CREATE CONSTRAINT category_id IF NOT EXISTS FOR (cat:Category) REQUIRE cat.id IS UNIQUE",
            "CREATE CONSTRAINT subcategory_id IF NOT EXISTS FOR (sub:Subcategory) REQUIRE sub.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
            
            # Create indexes for performance
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Diagnosis) ON (d.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Code) ON (c.code)")
            
            logger.info("Database schema setup complete")
    
    def add_icd10_code(self, tx, code: str, description: str, 
                      category: Optional[str] = None, 
                      subcategory: Optional[str] = None,
                      includes: Optional[List[str]] = None,
                      excludes: Optional[List[str]] = None,
                      notes: Optional[str] = None):
        """
        Add an ICD-10-CM code to the database with its relationships.
        
        Args:
            tx: Neo4j transaction
            code: The ICD-10-CM code
            description: Description of the diagnosis
            category: Category the code belongs to
            subcategory: Subcategory the code belongs to
            includes: List of inclusion notes
            excludes: List of exclusion notes
            notes: Additional notes for the code
        """
        # Create the Code node
        tx.run(
            """
            MERGE (c:Code {code: $code})
            ON CREATE SET c.description = $description, c.notes = $notes
            ON MATCH SET c.description = $description, c.notes = $notes
            """,
            code=code, description=description, notes=notes
        )
        
        # Create the Diagnosis node and relationship
        tx.run(
            """
            MERGE (d:Diagnosis {name: $description})
            MERGE (d)-[:MAPS_TO]->(c:Code {code: $code})
            """,
            description=description, code=code
        )
        
        # Add category if provided
        if category:
            tx.run(
                """
                MERGE (cat:Category {id: $category})
                MERGE (c:Code {code: $code})-[:BELONGS_TO]->(cat)
                """,
                category=category, code=code
            )
        
        # Add subcategory if provided
        if subcategory:
            tx.run(
                """
                MERGE (sub:Subcategory {id: $subcategory})
                MERGE (c:Code {code: $code})-[:BELONGS_TO]->(sub)
                MERGE (sub)-[:PART_OF]->(cat:Category {id: $category})
                """,
                subcategory=subcategory, category=category, code=code
            )
        
        # Add inclusion notes
        if includes:
            for include in includes:
                tx.run(
                    """
                    MERGE (i:InclusionNote {text: $include})
                    MERGE (c:Code {code: $code})-[:INCLUDES]->(i)
                    """,
                    include=include, code=code
                )
        
        # Add exclusion notes
        if excludes:
            for exclude in excludes:
                tx.run(
                    """
                    MERGE (e:ExclusionNote {text: $exclude})
                    MERGE (c:Code {code: $code})-[:EXCLUDES]->(e)
                    """,
                    exclude=exclude, code=code
                )
    
    def find_code_by_diagnosis(self, diagnosis: str) -> List[Dict]:
        """
        Find ICD-10-CM codes by diagnosis description.
        
        Args:
            diagnosis: The diagnosis text to search for
            
        Returns:
            List of matching codes with descriptions
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Diagnosis)-[:MAPS_TO]->(c:Code)
                WHERE d.name CONTAINS $term OR toLower(d.name) CONTAINS toLower($term)
                RETURN c.code AS code, c.description AS description
                LIMIT 10
                """,
                term=diagnosis
            )
            return [dict(record) for record in result]
    
    def find_code_by_code(self, code: str) -> Optional[Dict]:
        """
        Find an ICD-10-CM code by its code value.
        
        Args:
            code: The ICD-10-CM code to search for
            
        Returns:
            Code details if found, None otherwise
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Code {code: $code})
                OPTIONAL MATCH (c)-[:INCLUDES]->(i:InclusionNote)
                OPTIONAL MATCH (c)-[:EXCLUDES]->(e:ExclusionNote)
                RETURN 
                    c.code AS code, 
                    c.description AS description,
                    c.notes AS notes,
                    collect(DISTINCT i.text) AS includes,
                    collect(DISTINCT e.text) AS excludes
                """,
                code=code
            )
            record = result.single()
            return dict(record) if record else None
    
    def search_codes(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for ICD-10-CM codes using a text query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching codes with descriptions
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Code)
                WHERE c.code CONTAINS $query OR 
                      toLower(c.description) CONTAINS toLower($query)
                RETURN c.code AS code, c.description AS description
                LIMIT $limit
                """,
                query=query, limit=limit
            )
            return [dict(record) for record in result]
    
    def get_related_codes(self, code: str) -> List[Dict]:
        """
        Get codes related to the given code.
        
        Args:
            code: The ICD-10-CM code to find relations for
            
        Returns:
            List of related codes with descriptions and relationship types
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Code {code: $code})-[:BELONGS_TO]->(cat)
                MATCH (related:Code)-[:BELONGS_TO]->(cat)
                WHERE related.code <> $code
                RETURN related.code AS code, related.description AS description, 
                       "SAME_CATEGORY" AS relationship
                LIMIT 10
                """,
                code=code
            )
            return [dict(record) for record in result]