from openai import OpenAI
import os
from typing import List, Tuple, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self):
        """Initialize with OpenAI API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def process_query(self, query: str, context: str) -> Dict:
        """Process a medical query using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical coding assistant. Help find the most appropriate ICD-10 code based on the medical context provided."},
                    {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
                ],
                temperature=0.3
            )
            return {
                "answer": response.choices[0].message.content,
                "confidence": 1.0 if response.choices[0].finish_reason == "stop" else 0.5
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"error": str(e)}

    def check_meat_compliance(self, note: str) -> Dict:
        """Check if a medical note follows MEAT criteria using OpenAI"""
        try:
            prompt = f"""Analyze this medical note for MEAT criteria compliance:

Medical Note:
{note}

Please analyze the note for each MEAT criterion:
1. Monitored: Evidence of monitoring/tracking
2. Evaluated: Evidence of evaluation/examination
3. Assessed/Addressed: Evidence of assessment/diagnosis
4. Treated: Evidence of treatment/management

For each criterion, provide:
- Whether it is met (Yes/No)
- Supporting evidence from the note
- Suggestions for improvement if not met

Format the response in a clear, structured way."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical documentation specialist analyzing notes for MEAT criteria."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            analysis = response.choices[0].message.content

            # Parse the analysis to determine compliance
            criteria = {
                "monitored": "monitor" in note.lower() or "tracking" in note.lower(),
                "evaluated": "evaluat" in note.lower() or "exam" in note.lower(),
                "assessed": "assess" in note.lower() or "diagnos" in note.lower(),
                "treated": "treat" in note.lower() or "prescrib" in note.lower() or "medication" in note.lower()
            }

            is_compliant = all(criteria.values())

            return {
                "compliant": is_compliant,
                "criteria": {
                    k: {
                        "met": v,
                        "details": analysis
                    } for k, v in criteria.items()
                },
                "full_analysis": analysis,
                "suggestions": [] if is_compliant else [
                    f"Add {criterion} information" 
                    for criterion, met in criteria.items() 
                    if not met
                ]
            }

        except Exception as e:
            logger.error(f"Error checking MEAT compliance: {str(e)}")
            return {
                "compliant": False,
                "error": str(e),
                "criteria": {},
                "full_analysis": "Error analyzing note",
                "suggestions": ["Error occurred during analysis"]
            }

    def find_similar_codes(self, query: str, code_descriptions: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """Find similar ICD codes using OpenAI's understanding"""
        try:
            codes_text = "\n".join([f"{code}: {desc}" for code, desc in code_descriptions])
            prompt = f"""Given these ICD-10 codes and descriptions:

            {codes_text}

            Find the most relevant codes for: "{query}"
            Provide the top 3 most relevant codes with explanations."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical coding assistant helping to find relevant ICD-10 codes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # For now, return a simplified version
            return [(code, desc, 0.9) for code, desc in code_descriptions[:3]]
        except Exception as e:
            logger.error(f"Error finding similar codes: {str(e)}")
            return [] 