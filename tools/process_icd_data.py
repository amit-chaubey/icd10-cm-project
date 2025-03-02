import xml.etree.ElementTree as ET
import json
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_mainterm(main_term):
    """Process a main term element from XML"""
    result = {
        "title": main_term.find("title").text,
        "code": [code.text for code in main_term.findall("code")],
        "see": [see.text for see in main_term.findall("see")],
        "see_also": [ref.text for ref in main_term.findall("seeAlso")],
        "modifiers": [mod.text for mod in main_term.findall(".//nemod")],
        "term": []
    }
    
    # Process subterms
    for subterm in main_term.findall("term"):
        result["term"].append(process_term(subterm))
        
    return result

def process_term(term_elem):
    """Process a term element from XML"""
    result = {
        "title": term_elem.find("title").text,
        "level": term_elem.get("level"),
        "code": [code.text for code in term_elem.findall("code")],
        "see": [see.text for see in term_elem.findall("see")],
        "see_also": [ref.text for ref in term_elem.findall("seeAlso")],
        "modifiers": [mod.text for mod in term_elem.findall(".//nemod")],
        "term": []
    }
    
    # Process nested subterms
    for subterm in term_elem.findall("term"):
        result["term"].append(process_term(subterm))
        
    return result

def convert_xml_to_json(input_path: str, output_path: str):
    """Convert ICD-10 XML index to structured JSON"""
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    result = {
        "version": root.find("version").text,
        "title": root.find("title").text,
        "terms": {}
    }
    
    # Process each letter section
    for letter_elem in root.findall("letter"):
        letter = letter_elem.find("title").text
        result["terms"][letter] = {
            "terms": []
        }
        
        # Process main terms
        for term_elem in letter_elem.findall("mainTerm"):
            term_data = process_mainterm(term_elem)
            result["terms"][letter]["terms"].append(term_data)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    convert_xml_to_json(
        "data/processed/icd10cm-table-index-April-2025/icd10cm-index-April-2025.xml",
        "data/processed/indexed/icd10cm_index.json"
    )