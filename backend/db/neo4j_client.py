from neo4j import GraphDatabase
from typing import Optional, List, Dict, Any
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        """Initialize Neo4j client"""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD not set")
            
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password)
        )
        
    def close(self):
        """Close the Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()

    def verify_connection(self) -> bool:
        """Verify that the connection to Neo4j is working"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Connection verification failed: {str(e)}")
            return False

    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess the query to extract meaningful medical terms"""
        # Convert to lowercase
        query = query.lower()
        
        # Replace common number variations
        query = query.replace("type two", "type 2")
        query = query.replace("type ii", "type 2")
        
        # Remove common filler words
        filler_words = ["patient", "has", "with", "suffering", "from", "showing", "symptoms", "of", "the"]
        for word in filler_words:
            query = query.replace(word, " ")
        
        # Split into words and remove empty strings
        words = [w.strip() for w in query.split() if w.strip()]
        
        # Generate combinations of consecutive words
        terms = []
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                term = " ".join(words[i:j])
                if len(term) > 2:  # Only include terms longer than 2 characters
                    terms.append(term)
        
        return terms

    def query_diagnosis(self, term: str) -> List[Dict]:
        """Query diagnosis terms and codes"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Term)
                WHERE toLower(t.name) CONTAINS toLower($term)
                OR any(modifier IN t.modifiers WHERE toLower(modifier) CONTAINS toLower($term))
                OPTIONAL MATCH (t)-[:HAS_CODE]->(c:Code)
                OPTIONAL MATCH (t)-[:SEE]->(s:Term)
                OPTIONAL MATCH (t)-[:SEE_ALSO]->(sa:Term)
                RETURN DISTINCT t.name as term,
                       t.level as level,
                       collect(DISTINCT c.value) as codes,
                       collect(DISTINCT sa.name) as see_also,
                       collect(DISTINCT s.name) as see,
                       t.modifiers as modifiers
                ORDER BY size(t.name)
                LIMIT 5
            """, term=term)
            
            return [dict(record) for record in result]

    def get_similar_diagnoses(self, term: str, limit: int = 20, offset: int = 0) -> List[tuple]:
        """
        Get similar diagnoses with pagination support
        
        Args:
            term: Search term or letter prefix (if starts with ^)
            limit: Number of results per page
            offset: Number of results to skip
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (t:Term)
                WHERE (
                    CASE 
                        WHEN $term STARTS WITH '^' 
                        THEN t.name STARTS WITH right($term, size($term)-1)
                        ELSE toLower(t.name) CONTAINS toLower($term)
                    END
                )
                WITH t
                ORDER BY t.name
                SKIP $offset
                LIMIT $limit
                OPTIONAL MATCH (t)-[:HAS_CODE]->(c:Code)
                RETURN t.name as term, 
                       collect(DISTINCT c.value) as codes,
                       t.modifiers as modifiers
                """
                
                result = session.run(
                    query,
                    term=term,
                    limit=limit,
                    offset=offset
                )
                
                return [(r["term"], r["codes"] or [], r.get("modifiers", [])) 
                        for r in result]
                        
        except Exception as e:
            logger.error(f"Error getting similar diagnoses: {str(e)}")
            return []

    def get_all_letters(self):
        """Get all available letter categories"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (l:Letter)
                    RETURN l.name as letter
                    ORDER BY l.name
                    """
                )
                return [record["letter"] for record in result]
        except Exception as e:
            logger.error(f"Error getting letters: {str(e)}")
            return []

    def create_constraints(self):
        """Create enhanced database constraints and indices"""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Letter) REQUIRE l.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Code) REQUIRE c.value IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.id IS UNIQUE",
                "CREATE INDEX term_name IF NOT EXISTS FOR (t:Term) ON (t.name)",
                "CREATE INDEX term_path IF NOT EXISTS FOR (t:Term) ON (t.path)",
                "CREATE INDEX code_category IF NOT EXISTS FOR (c:Code) ON (c.category)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint/index: {constraint}")
                except Exception as e:
                    logger.error(f"Error creating constraint: {str(e)}")
                    raise

    def load_icd_data(self, data: Dict[str, Any]):
        """Load ICD data with hierarchical structure into Neo4j"""
        try:
            with self.driver.session() as session:
                # First clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared existing database")
                
                # Create version node
                session.run("""
                    CREATE (v:Version {number: $version, title: $title})
                """, version=data['version'], title=data['title'])
                logger.info(f"Created version node: {data['version']}")
                
                # Process each letter and its terms
                for letter, letter_data in data['terms'].items():
                    logger.info(f"Processing letter: {letter}")
                    self._load_letter_terms(session, letter, letter_data['terms'])
                    
                logger.info("Successfully loaded ICD-10-CM data")
                
        except Exception as e:
            logger.error(f"Error loading ICD data: {str(e)}")
            raise

    def _load_letter_terms(self, session, letter: str, terms: List[Dict]):
        """Load terms for a specific letter"""
        # Create letter node
        session.run("""
            MERGE (l:Letter {name: $letter})
        """, letter=letter)
        
        # Process each main term
        for term_data in terms:
            self._create_term_hierarchy(session, term_data, letter)
            
    def _create_term_hierarchy(self, session, term_data: Dict, letter: str, parent_id=None):
        """Recursively create term hierarchy"""
        # Create term node with all properties
        result = session.run("""
            CREATE (t:Term {
                name: $name,
                codes: $codes,
                see: $see,
                see_also: $see_also,
                modifiers: $modifiers
            })
            RETURN id(t) as term_id
        """, 
            name=term_data['term'],
            codes=term_data.get('codes', []),
            see=term_data.get('see', []),
            see_also=term_data.get('see_also', []),
            modifiers=term_data.get('modifiers', [])
        )
        
        term_id = result.single()["term_id"]
        
        # Create relationship to letter
        session.run("""
            MATCH (l:Letter {name: $letter})
            MATCH (t:Term) WHERE id(t) = $term_id
            CREATE (l)-[:CONTAINS]->(t)
        """, letter=letter, term_id=term_id)
        
        # Create relationship to parent if exists
        if parent_id:
            session.run("""
                MATCH (parent:Term) WHERE id(parent) = $parent_id
                MATCH (child:Term) WHERE id(child) = $term_id
                CREATE (parent)-[:HAS_SUBTERM]->(child)
            """, parent_id=parent_id, term_id=term_id)
        
        # Create code nodes and relationships
        for code in term_data.get('codes', []):
            session.run("""
                MERGE (c:Code {value: $code})
                WITH c
                MATCH (t:Term) WHERE id(t) = $term_id
                CREATE (t)-[:HAS_CODE]->(c)
            """, code=code, term_id=term_id)
        
        # Process subterms recursively
        for subterm in term_data.get('subTerms', []):
            self._create_term_hierarchy(session, subterm, letter, term_id)

    def create_term_hierarchy(self, session, term_data: Dict, letter: str, parent_id=None, parent_path=""):
        """Create term hierarchy in Neo4j with improved structure"""
        try:
            # Generate unique ID and path
            term_name = term_data.get('title', '')
            term_path = f"{parent_path}/{term_name}" if parent_path else f"{letter}/{term_name}"
            term_id = f"{letter}_{term_name}_{term_data.get('level', 'main')}".replace(" ", "_").replace("/", "_")

            # Create term node with enhanced properties using elementId
            result = session.run("""
                CREATE (t:Term {
                    term_id: $term_id,
                    name: $name,
                    level: $level,
                    modifiers: $modifiers,
                    see: $see,
                    see_also: $see_also,
                    path: $path
                })
                RETURN elementId(t) as node_id
            """,
                term_id=term_id,
                name=term_name,
                level=term_data.get('level', 'main'),
                modifiers=term_data.get('modifiers', []),
                see=term_data.get('see', []),
                see_also=term_data.get('see_also', []),
                path=term_path
            )
            
            node_id = result.single()["node_id"]
            
            # Create relationships
            self._create_enhanced_relationships(
                session, 
                node_id, 
                term_data, 
                letter, 
                parent_id,
                term_path
            )
            
            return node_id
            
        except Exception as e:
            logger.error(f"Error creating term: {term_name}")
            logger.error(f"Error details: {str(e)}")
            raise

    def _create_enhanced_relationships(self, session, node_id: str, term_data: Dict, 
                                    letter: str, parent_id=None, term_path=""):
        """Create enhanced relationships for terms"""
        # Link to letter using elementId
        session.run("""
            MATCH (l:Letter {name: $letter})
            MATCH (t:Term) WHERE elementId(t) = $node_id
            CREATE (l)-[:CONTAINS {path: $path}]->(t)
        """, letter=letter, node_id=node_id, path=term_path)
        
        # Link to parent with path information
        if parent_id:
            session.run("""
                MATCH (p:Term) WHERE elementId(p) = $parent_id
                MATCH (c:Term) WHERE elementId(c) = $child_id
                CREATE (p)-[:HAS_SUBTERM {path: $path}]->(c)
            """, parent_id=parent_id, child_id=node_id, path=term_path)
        
        # Create code relationships with validation
        codes = term_data.get('code', [])
        if isinstance(codes, str):
            codes = [codes]
        
        for code in codes:
            if self._is_valid_icd_code(code):
                session.run("""
                    MERGE (c:Code {
                        value: $code,
                        category: $category,
                        subcategory: $subcategory
                    })
                    WITH c
                    MATCH (t:Term) WHERE elementId(t) = $node_id
                    CREATE (t)-[:HAS_CODE]->(c)
                """, 
                    code=code,
                    category=code.split('.')[0],
                    subcategory=code.split('.')[1] if '.' in code else '',
                    node_id=node_id
                )
        
        # Create cross-references with validation
        for see in term_data.get('see', []):
            session.run("""
                MERGE (ref:Term {name: $ref_name})
                WITH ref
                MATCH (t:Term) WHERE elementId(t) = $node_id
                CREATE (t)-[:SEE {type: 'direct'}]->(ref)
            """, ref_name=see, node_id=node_id)
            
        for see_also in term_data.get('see_also', []):
            session.run("""
                MERGE (ref:Term {name: $ref_name})
                WITH ref
                MATCH (t:Term) WHERE elementId(t) = $node_id
                CREATE (t)-[:SEE_ALSO {type: 'related'}]->(ref)
            """, ref_name=see_also, node_id=node_id)

    def _is_valid_icd_code(self, code: str) -> bool:
        """Validate ICD-10 code format"""
        pattern = r'^[A-Z][0-9][0-9A-Z]\.?[0-9A-Z]{0,4}$'
        return bool(re.match(pattern, code))

    def add_medical_data(self, diagnosis: str, code: str):
        """Add medical diagnosis and code to the database"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Diagnosis {name: $diagnosis}) 
                MERGE (c:Code {value: $code}) 
                MERGE (d)-[:MAPS_TO]->(c)
                """,
                diagnosis=diagnosis,
                code=code
            )

    def clear_database(self) -> bool:
        """Clear all data from the database"""
        try:
            with self.driver.session() as session:
                # Drop existing constraints
                constraints = session.run("SHOW CONSTRAINTS").data()
                for constraint in constraints:
                    constraint_name = constraint.get('name')
                    if constraint_name:
                        session.run(f"DROP CONSTRAINT {constraint_name}")
                        logger.info(f"Dropped constraint: {constraint_name}")

                # Remove all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared all nodes and relationships")
                return True
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False

    def query_by_letter(self, letter: str) -> List[Dict]:
        """Query terms by starting letter"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (l:Letter {name: $letter})-[:CONTAINS]->(t:Term)
                OPTIONAL MATCH (t)-[:HAS_CODE]->(c:Code)
                RETURN DISTINCT t.name as term,
                       t.level as level,
                       collect(DISTINCT c.value) as codes,
                       t.modifiers as modifiers
                ORDER BY t.name
                LIMIT 25
            """, letter=letter)
            
            return [dict(record) for record in result]