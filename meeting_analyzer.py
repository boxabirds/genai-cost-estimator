import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MeetingAnalyzer:
    def __init__(self, api_key: str, price_file: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.prices = self._load_prices(price_file)
        
    def _load_prices(self, price_file: str) -> Dict:
        with open(price_file, 'r') as f:
            prices = json.load(f)
        # Find Gemini 1.5 Pro pricing
        for model in prices:
            if model["Model"] == "Gemini 1.5 Pro":
                return {
                    "input": float(model["Input"]),
                    "output": float(model["Output"])
                }
        raise ValueError("Gemini 1.5 Pro pricing not found in price file")

    def analyze_meeting(self, transcript_file: str) -> Tuple[Dict, Dict]:
        # Read transcript
        with open(transcript_file, 'r') as f:
            transcript = f.read()

        # Split into chunks if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=30000,  # Gemini can handle larger chunks
            chunk_overlap=1000,
            length_function=len,
        )
        chunks = text_splitter.split_text(transcript)

        # Analysis prompt
        prompt_template = """Analyze the following meeting transcript and provide:
        1. A clear and concise summary of the main points discussed
        2. Key decisions or actions agreed upon, including who is responsible and any deadlines
        3. What went well in the meeting
        4. What could be improved about the meeting

        Transcript:
        {transcript}

        Please format your response in JSON with these keys:
        - summary
        - actions
        - positives
        - improvements"""

        # Process each chunk and combine results
        all_results = []
        total_tokens = {"input": 0, "output": 0}
        
        for chunk in chunks:
            # Process with Gemini
            prompt = prompt_template.format(transcript=chunk)
            response = self.model.generate_content(prompt)
            
            # Get token counts from usage metadata
            if hasattr(response, 'usage_metadata'):
                total_tokens["input"] += response.usage_metadata.prompt_token_count
                total_tokens["output"] += response.usage_metadata.candidates_token_count
            
            # Parse result
            try:
                parsed_result = json.loads(response.text)
                all_results.append(parsed_result)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse LLM output as JSON: {response.text[:100]}...")

        # Combine results
        combined_results = {
            "summary": "\n".join(r.get("summary", "") for r in all_results),
            "actions": [action for r in all_results for action in r.get("actions", [])],
            "positives": [pos for r in all_results for pos in r.get("positives", [])],
            "improvements": [imp for r in all_results for imp in r.get("improvements", [])]
        }

        # Calculate costs (using direct token counts)
        costs = {
            "input_cost": (total_tokens["input"] / 1000) * self.prices["input"],
            "output_cost": (total_tokens["output"] / 1000) * self.prices["output"],
            "total_cost": ((total_tokens["input"] / 1000) * self.prices["input"]) + 
                         ((total_tokens["output"] / 1000) * self.prices["output"])
        }

        return combined_results, costs

def main():
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")

    # Initialize analyzer
    analyzer = MeetingAnalyzer(
        api_key=api_key,
        price_file="/Users/julian/expts/genai-cost-estimator/llm-prices-20241120.json"
    )

    # Analyze meeting
    transcript_file = "/Users/julian/expts/genai-cost-estimator/data/meeting-transcript-1.md"
    results, costs = analyzer.analyze_meeting(transcript_file)

    # Print results
    print("\n=== Meeting Analysis ===")
    print("\nSummary:")
    print(results["summary"])
    
    print("\nActions Agreed:")
    for action in results["actions"]:
        print(f"- {action}")
    
    print("\nWhat Went Well:")
    for positive in results["positives"]:
        print(f"- {positive}")
    
    print("\nAreas for Improvement:")
    for improvement in results["improvements"]:
        print(f"- {improvement}")
    
    print("\n=== Cost Analysis ===")
    print(f"Input cost: ${costs['input_cost']:.4f}")
    print(f"Output cost: ${costs['output_cost']:.4f}")
    print(f"Total cost: ${costs['total_cost']:.4f}")

if __name__ == "__main__":
    main()
