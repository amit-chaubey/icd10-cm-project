"""
Streamlit UI for the ICD-10-CM Coding Assistant.
"""
import sys
import os
from pathlib import Path
import logging
import streamlit as st
import pandas as pd
from io import StringIO
import json
from typing import Dict, List, Any
import openai

# Configure page first, before any other st commands
st.set_page_config(
    page_title="ICD-10-CM Coding Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the project root directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Now we can import from backend
from backend.db.neo4j_client import Neo4jClient
from backend.models.llm_processor import LLMProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def search_diagnosis(neo4j_client, query):
    """Perform the diagnosis search and display results"""
    try:
        with st.spinner("Searching..."):
            results = neo4j_client.query_diagnosis(query)
            if results:
                st.success("Found matching diagnoses:")
                display_search_results(results)
            else:
                st.warning("No exact match found. Try browsing by letter.")
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        st.error("An error occurred during search. Please try again.")

def display_search_results(results):
    """Display search results with pagination"""
    ITEMS_PER_PAGE = 10
    
    if not results:
        st.warning("No matches found. Try different search terms.")
        return
    
    # Add pagination
    total_pages = len(results) // ITEMS_PER_PAGE + (1 if len(results) % ITEMS_PER_PAGE > 0 else 0)
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1))
    else:
        page = 1
    
    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    page_results = results[start_idx:end_idx]
    
    for result in page_results:
        with st.expander(f"📋 {result['term']}", expanded=len(page_results) == 1):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if result.get('codes'):
                    st.markdown("### ICD-10 Codes")
                    st.code("\n".join(result['codes']), language="text")
                
                if result.get('modifiers') and any(result['modifiers']):
                    st.markdown("### Modifiers")
                    st.write("\n".join(f"- {mod}" for mod in result['modifiers']))
            
            with col2:
                if result.get('see_also') and any(result['see_also']):
                    st.markdown("### See Also")
                    st.write("\n".join(f"- {ref}" for ref in result['see_also']))

def display_similar_results(similar_results):
    """Display similar diagnosis results"""
    if not similar_results:
        st.info("No similar codes found.")
        return
        
    for term, codes, modifiers in similar_results:
        with st.expander(f"🔍 {term}"):
            if codes:
                st.markdown("### ICD-10 Codes")
                for code in codes:
                    st.code(code, language="text")
            
            if modifiers:
                st.markdown("### Modifiers")
                for mod in modifiers:
                    st.write(f"- {mod}")

def format_diagnosis_result(result: dict) -> None:
    """Format and display a single diagnosis result"""
    with st.expander(f"📋 {result['term']}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if result['codes']:
                st.markdown("### ICD-10 Codes")
                for code in result['codes']:
                    st.code(f"{code}", language="text")
            
            if result.get('modifiers'):
                st.markdown("### Modifiers")
                for mod in result['modifiers']:
                    st.write(f"- {mod}")
            
            if result.get('includes'):
                st.markdown("### Includes")
                for inc in result['includes']:
                    st.write(f"- {inc}")
                    
            if result.get('excludes'):
                st.markdown("### Excludes")
                for exc in result['excludes']:
                    st.write(f"- {exc}")
        
        with col2:
            if result.get('see_also'):
                st.markdown("### See Also")
                for see in result['see_also']:
                    st.write(f"- {see}")
            
            if result.get('sub_terms'):
                st.markdown("### Related Terms")
                for sub in result['sub_terms']:
                    st.write(f"- {sub}")

def process_clinical_note(note_text, neo4j_client):
    """Process a clinical note and extract relevant information using LLM"""
    llm = LLMProcessor()
    
    try:
        # Extract medical conditions and context using GPT
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a medical coding assistant. 
                Extract key medical conditions, symptoms, and relevant clinical context. 
                Format the response as JSON with the following structure:
                {
                    "conditions": [{"condition": "", "context": "", "severity": ""}],
                    "symptoms": [],
                    "procedures": [],
                    "medications": []
                }"""},
                {"role": "user", "content": note_text}
            ],
            temperature=0.3
        )
        
        # Parse extracted information
        extracted_info = json.loads(response.choices[0].message.content)
        
        # Check MEAT compliance
        meat_analysis = llm.check_meat_compliance(note_text)
        
        # Query Neo4j for ICD codes for each condition
        icd_codes = []
        for condition in extracted_info['conditions']:
            codes = neo4j_client.query_diagnosis(condition['condition'])
            if codes:
                icd_codes.extend(codes)
        
        return {
            "extracted_info": extracted_info,
            "meat_compliance": meat_analysis,
            "icd_codes": icd_codes
        }
        
    except Exception as e:
        logger.error(f"Error processing clinical note: {e}")
        return None

def process_medical_query(self, query: str) -> Dict[str, Any]:
    """Process medical query using OpenAI to extract searchable terms"""
    try:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical coding assistant. Extract key medical terms from the query."},
                {"role": "user", "content": f"Extract key medical terms from: {query}"}
            ],
            temperature=0.3
        )
        extracted_terms = response.choices[0].message.content
        return {
            "original_query": query,
            "extracted_terms": extracted_terms,
            "search_terms": [term.strip() for term in extracted_terms.split(',')]
        }
    except Exception as e:
        logger.error(f"Error processing medical query: {str(e)}")
        return {"main_condition": query, "related_terms": [], "modifiers": []}

def init_neo4j():
    """Initialize Neo4j client with connection verification"""
    logger.info("Initializing Neo4j client...")
    try:
        client = Neo4jClient()
        logger.info("Testing Neo4j connection...")
        if client.verify_connection():
            logger.info("Successfully connected to Neo4j")
            return client
        else:
            logger.error("Failed to verify Neo4j connection")
            st.error("❌ Could not connect to Neo4j database")
            return None
    except Exception as e:
        logger.error(f"Neo4j initialization error: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize Neo4j client: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_cached_letters(_neo4j_client):
    """Cache letters to avoid repeated database calls"""
    try:
        return _neo4j_client.get_all_letters()
    except Exception as e:
        logger.error(f"Error fetching letters: {str(e)}")
        return []

def show_connection_error():
    st.error("Database connection error!")
    st.write("Please ensure:")
    st.write("1. Neo4j is running (`docker ps` to check)")
    st.write("2. Credentials in .env file are correct")
    st.write("3. You can connect to Neo4j browser at http://localhost:7474")
    
    if st.button("Try Reconnecting"):
        st.experimental_rerun()

def display_meat_compliance(results):
    """Display MEAT compliance results in a user-friendly format"""
    if not results or "meat_compliance" not in results:
        st.error("No MEAT compliance results available")
        return

    meat = results["meat_compliance"]
    
    # Overall compliance status
    if meat["compliant"]:
        st.success("✅ Note is MEAT compliant")
    else:
        st.warning("⚠️ Note needs improvement to meet MEAT criteria")

    # Display detailed analysis
    st.subheader("Detailed MEAT Analysis")
    
    # Create columns for each criterion
    cols = st.columns(4)
    criteria_icons = {
        "monitored": "📊",
        "evaluated": "🔍",
        "assessed": "📋",
        "treated": "💊"
    }
    
    for col, (criterion, details) in zip(cols, meat["criteria"].items()):
        with col:
            st.markdown(f"### {criteria_icons[criterion]} {criterion.title()}")
            if details["met"]:
                st.success("✓ Criterion Met")
            else:
                st.error("✗ Not Found")

    # Display full analysis
    with st.expander("View Full Analysis"):
        st.markdown(meat["full_analysis"])

    # Show suggestions if any
    if not meat["compliant"] and "suggestions" in meat:
        st.subheader("📝 Suggestions for Improvement")
        for suggestion in meat["suggestions"]:
            st.info(suggestion)

def display_clinical_analysis(results):
    """Display comprehensive clinical note analysis"""
    if not results:
        st.error("Failed to analyze the clinical note")
        return
        
    # Display extracted information
    st.subheader("📋 Extracted Clinical Information")
    
    # Display conditions and their ICD codes
    if results['extracted_info']['conditions']:
        st.markdown("### Medical Conditions")
        for condition in results['extracted_info']['conditions']:
            with st.expander(f"🏥 {condition['condition']}", expanded=True):
                st.markdown(f"**Context:** {condition['context']}")
                if condition.get('severity'):
                    st.markdown(f"**Severity:** {condition['severity']}")
                
                # Display relevant ICD codes
                matching_codes = [
                    code for code in results['icd_codes'] 
                    if condition['condition'].lower() in code['term'].lower()
                ]
                if matching_codes:
                    st.markdown("#### Related ICD-10 Codes:")
                    for code in matching_codes:
                        st.code(f"{code['code']} - {code['term']}")
    
    # Display other extracted information
    if results['extracted_info'].get('symptoms'):
        st.markdown("### Symptoms")
        for symptom in results['extracted_info']['symptoms']:
            st.markdown(f"- {symptom}")
            
    if results['extracted_info'].get('procedures'):
        st.markdown("### Procedures")
        for procedure in results['extracted_info']['procedures']:
            st.markdown(f"- {procedure}")
            
    if results['extracted_info'].get('medications'):
        st.markdown("### Medications")
        for medication in results['extracted_info']['medications']:
            st.markdown(f"- {medication}")
    
    # Display MEAT compliance analysis
    st.markdown("---")
    display_meat_compliance(results)

def get_icd_codes(query: str, neo4j_client) -> dict:
    """Get ICD codes with LLM assistance"""
    try:
        # First, let LLM clean and extract the key medical terms
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical coding expert. Extract the main medical condition from the input."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        
        cleaned_term = response.choices[0].message.content
        
        # Query Neo4j for ICD codes
        codes = neo4j_client.query_diagnosis(cleaned_term)
        
        # Let LLM analyze and explain the codes
        if codes:
            code_explanation = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical coding expert. Explain which ICD code best matches the condition and why."},
                    {"role": "user", "content": f"Condition: {cleaned_term}\nAvailable codes: {codes}"}
                ],
                temperature=0.3
            )
            
            return {
                "original_query": query,
                "extracted_term": cleaned_term,
                "codes": codes,
                "explanation": code_explanation.choices[0].message.content
            }
        return None
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return None

def display_code_search_results(results: List[Dict[str, Any]]):
    """Display ICD code search results in a formatted way"""
    if not results:
        st.warning("No results found. Try different search terms.")
        return
        
    for result in results:
        with st.expander(f"🔍 {result.get('term', 'Unknown Term')}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display ICD Codes
                if result.get('codes'):
                    st.markdown("### ICD-10 Codes")
                    for code in result['codes']:
                        st.code(f"{code}", language="text")
                
                # Display Modifiers
                if result.get('modifiers') and any(result['modifiers']):
                    st.markdown("### Modifiers")
                    for mod in result['modifiers']:
                        st.markdown(f"- {mod}")
            
            with col2:
                # Display See References
                if result.get('see') and any(result['see']):
                    st.markdown("### See")
                    for see in result['see']:
                        st.markdown(f"- {see}")
                
                # Display See Also References
                if result.get('see_also') and any(result['see_also']):
                    st.markdown("### See Also")
                    for see_also in result['see_also']:
                        st.markdown(f"- {see_also}")

def process_clinical_terms(clinical_note: str) -> List[str]:
    """Extract meaningful medical terms from clinical note"""
    
    # Key medical sections to focus on
    key_sections = ['assessment:', 'impression:', 'diagnosis:', 'diagnoses:']
    
    # Enhanced exclusion lists
    exclude_words = {
        'patient', 'reports', 'states', 'with', 'and', 'the', 'has', 'had',
        'shows', 'revealed', 'demonstrates', 'presents', 'complains', 'notes',
        'denies', 'following', 'started', 'given', 'received', 'current',
        'active', 'ongoing', 'daily', 'weekly', 'due', 'to', 'at', 'on', 'in'
    }

    medical_stop_words = {
        'status', 'condition', 'normal', 'abnormal', 'positive', 'negative',
        'history', 'past', 'family', 'social', 'medication', 'medications',
        'vital', 'vitals', 'signs', 'sign', 'lab', 'labs', 'test', 'tests'
    }

    MIN_TERM_LENGTH = 3
    MAX_TERM_LENGTH = 6  # Maximum words in a term
    
    terms = []
    
    # Split into sections
    sections = [s.strip().lower() for s in clinical_note.split('\n') if s.strip()]
    
    # Prioritize assessment/impression sections
    assessment_section = False
    for section in sections:
        if any(key in section.lower() for key in key_sections):
            assessment_section = True
            continue
            
        if assessment_section:
            # Process numbered or bulleted diagnoses
            if ':' in section or section.startswith(('•', '-', '*', '1.', '2.', '3.')):
                cleaned_term = section.split(':', 1)[-1].strip()
                cleaned_term = ' '.join(w for w in cleaned_term.split() 
                                      if w.lower() not in exclude_words)
                if cleaned_term:
                    terms.append(cleaned_term)
    
    # Process regular sections if no assessment section found
    if not terms:
        for section in sections:
            words = section.split()
            for i in range(len(words)):
                for j in range(i+1, min(i+MAX_TERM_LENGTH, len(words)+1)):
                    phrase = ' '.join(words[i:j])
                    words_in_phrase = phrase.lower().split()
                    
                    if (len(phrase) >= MIN_TERM_LENGTH and
                        not all(word in exclude_words for word in words_in_phrase) and
                        not all(word in medical_stop_words for word in words_in_phrase) and
                        not any(char.isdigit() for char in phrase)):
                        terms.append(phrase.strip())
    
    # Remove duplicates and sort by specificity
    terms = sorted(set(terms), key=len, reverse=True)
    
    # Filter out substrings that are part of longer terms
    filtered_terms = []
    for i, term in enumerate(terms):
        if not any(term in other_term 
                  for other_term in terms[:i] + terms[i+1:]):
            filtered_terms.append(term)
    
    return filtered_terms[:10]  # Limit to top 10 most relevant terms

def perform_search(neo4j_client, search_query):
    """Execute search and display results"""
    logger.info(f"Searching for: {search_query}")
    
    try:
        with st.spinner("Searching..."):
            # Log before query
            logger.info("Querying Neo4j database...")
            
            results = neo4j_client.query_diagnosis(search_query)
            
            # Log results
            logger.info(f"Found {len(results) if results else 0} results")
            
            if results:
                st.success(f"Found {len(results)} matching codes")
                # Display results in collapsed expanders
                for result in results:
                    logger.debug(f"Processing result: {result}")
                    with st.expander(f"🔍 {result['term']}", expanded=False):
                        if result.get('codes'):
                            st.markdown("**ICD-10 Codes:**")
                            for code in result['codes']:
                                st.code(code)
                        if result.get('modifiers'):
                            st.markdown("**Modifiers:**")
                            for mod in result['modifiers']:
                                st.markdown(f"- {mod}")
            else:
                st.warning("No exact matches found")
                # Try finding similar terms
                logger.info("Attempting to find similar terms...")
                similar_results = neo4j_client.get_similar_diagnoses(search_query)
                if similar_results:
                    st.info("You might be interested in:")
                    for term, codes, _ in similar_results[:5]:
                        st.write(f"- {term}")
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        st.error(f"An error occurred during search: {str(e)}")

def main():
    # Main header
    st.title("🏥 ICD-10-CM Coding Assistant")
    st.markdown("---")
    
    # Initialize Neo4j client
    neo4j_client = init_neo4j()
    if not neo4j_client:
        st.error("❌ Failed to connect to database. Please check your connection.")
        return

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔍 Quick Code Lookup", "📚 Browse by Letter", "📝 Clinical Analysis"])
    
    # Tab 1: Quick Code Lookup
    with tab1:
        st.markdown("### Search for ICD-10 Codes")
        
        # Create a form to capture Enter key
        with st.form(key='search_form'):
            search_query = st.text_input(
                "Enter medical condition or term:",
                placeholder="e.g., Type 2 diabetes, Hypertension, Asthma...",
                key="search_input"
            )
            
            # Add search button
            submitted = st.form_submit_button("Search", type="primary")
            
            # Debug information
            if submitted:
                logger.info(f"Form submitted with query: {search_query}")
                
            # Show processing status
            if submitted and search_query:
                st.info("Processing your search...")
                perform_search(neo4j_client, search_query)
            elif submitted and not search_query:
                st.warning("Please enter a search term")

    # Tab 2: Browse by Letter
    with tab2:
        st.markdown("### Browse by Letter")
        letters = ["Select a letter..."] + (neo4j_client.get_all_letters() or [])
        selected_letter = st.selectbox(
            "Select a letter to browse terms:",
            letters,
            index=0
        )
        
        if selected_letter and selected_letter != "Select a letter...":
            with st.spinner(f"Loading terms for '{selected_letter}'..."):
                results = neo4j_client.query_by_letter(selected_letter)
                if results:
                    st.success(f"Found {len(results)} terms")
                    for result in results:
                        with st.expander(f"🔍 {result['term']}", expanded=False):
                            if result.get('codes'):
                                st.markdown("**ICD-10 Codes:**")
                                for code in result['codes']:
                                    st.code(code)
                            if result.get('modifiers'):
                                st.markdown("**Modifiers:**")
                                for mod in result['modifiers']:
                                    st.markdown(f"- {mod}")

    # Tab 3: Clinical Analysis
    with tab3:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Clinical Note Analysis")
            clinical_note = st.text_area(
                "Enter clinical note:",
                height=200,
                placeholder="Enter the patient's clinical note here..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                analyze = st.button("Analyze Note", type="primary")
            with col2:
                clear = st.button("Clear", type="secondary")
                
            if analyze and clinical_note:
                with st.spinner("Analyzing clinical note..."):
                    # Extract meaningful terms
                    terms = process_clinical_terms(clinical_note)
                    
                    # Group results by category
                    findings = {
                        'Diagnoses': [],
                        'Symptoms': [],
                        'Procedures': []
                    }
                    
                    # Track processed terms to avoid duplicates
                    processed_terms = set()
                    
                    for term in terms:
                        if term not in processed_terms:
                            results = neo4j_client.query_diagnosis(term)
                            if results:
                                for result in results:
                                    # Check code prefixes for categorization
                                    codes = result.get('codes', [])
                                    if codes:
                                        if any(code.startswith(('R', 'S', 'T')) for code in codes):
                                            findings['Symptoms'].append(result)
                                        elif any(code.startswith(('0', '1', '2', '3', '4')) for code in codes):
                                            findings['Procedures'].append(result)
                                        else:
                                            findings['Diagnoses'].append(result)
                                        processed_terms.add(term)
                    
                    # Display organized results
                    for category, results in findings.items():
                        if results:
                            st.markdown(f"### {category}")
                            # Remove duplicates based on term names
                            unique_results = {r['term']: r for r in results}.values()
                            for result in unique_results:
                                with st.expander(f"🔍 {result['term']}", expanded=False):
                                    if result.get('codes'):
                                        st.markdown("**ICD-10 Codes:**")
                                        for code in result['codes']:
                                            st.code(code)
                                    if result.get('modifiers'):
                                        st.markdown("**Modifiers:**")
                                        for mod in result['modifiers']:
                                            st.markdown(f"- {mod}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>ICD-10-CM Coding Assistant | v1.0.0</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()