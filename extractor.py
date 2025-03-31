import os
import re
import json
import argparse
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import requests
from pypdf import PdfReader
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BalanceSheetExtractor:
    """Extract financial data from balance sheet PDFs with LLM assistance."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate"):
        """
        Initialize the extractor.
        
        Args:
            ollama_url: URL for Ollama API
        """
        self.ollama_url = ollama_url
        self.model = "gemma3:latest" # llama3.2-vision:latest   |||||     Default model, can be changed based on availability
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            The extracted text as a string
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                    
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def query_ollama(self, prompt: str) -> str:
        """
        Query the Ollama API with a prompt.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            The model's response
        """
        logger.info(f"Querying Ollama with prompt length: {len(prompt)}")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info("Successfully received response from Ollama")
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            if "Connection refused" in str(e):
                logger.error("Make sure Ollama is running locally with 'ollama serve'")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize the extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'(\r\n|\r|\n)', '\n', text)
        
        # Remove page numbers and footers (common in financial statements)
        text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text)
        
        return text.strip()
    
    def extract_balance_sheet_structure(self, text: str) -> Dict:
        """
        Use Ollama to extract and structure balance sheet data.
        
        Args:
            text: Preprocessed text from the balance sheet
            
        Returns:
            Dictionary containing structured balance sheet data
        """
        # Prepare prompt for structure extraction
        prompt = f"""
        You are a financial data extraction assistant. Below is text from a balance sheet PDF.
        Please extract all financial information and format it as a clean JSON with the following structure:
        
        {{
            "company_name": "Company name if found",
            "statement_date": "Date of the balance sheet if found",
            "assets": {{
                "current_assets": {{
                    "cash_and_equivalents": value,
                    "accounts_receivable": value,
                    ...other current assets
                }},
                "non_current_assets": {{
                    "property_plant_equipment": value,
                    ...other non-current assets
                }},
                "total_assets": value
            }},
            "liabilities": {{
                "current_liabilities": {{
                    "accounts_payable": value,
                    ...other current liabilities
                }},
                "non_current_liabilities": {{
                    "long_term_debt": value,
                    ...other non-current liabilities
                }},
                "total_liabilities": value
            }},
            "equity": {{
                "common_stock": value,
                "retained_earnings": value,
                ...other equity items
                "total_equity": value
            }}
        }}
        
        Convert all numeric values to numbers, not strings. Only include items that actually appear in the balance sheet.
        Do not make up or infer values that are not explicitly stated. Use the exact terms from the balance sheet when possible.
        
        Here's the balance sheet text:
        
        {text}
        
        JSON output:
        """
        
        # Query Ollama for extraction
        try:
            response = self.query_ollama(prompt)
            
            # Extract JSON from response (the LLM might add commentary)
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code block markers
                json_match = re.search(r'(\{[\s\S]*\})', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response
            
            # Parse JSON response
            try:
                result = json.loads(json_str)
                logger.info("Successfully parsed balance sheet structure")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                logger.error(f"Raw response: {response}")
                raise ValueError("Ollama didn't return properly formatted JSON")
                
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            raise
    
    def extract_comparative_data(self, text: str) -> Dict:
        """
        Extract comparative data if multiple periods are present.
        
        Args:
            text: Preprocessed text from the balance sheet
            
        Returns:
            Dictionary with time period data
        """
        prompt = f"""
        You are a financial data extraction assistant. Analyze the balance sheet text below and identify if it contains data for multiple time periods.
        If so, extract the data for each period separately. Return the result as JSON with period identifiers.
        
        Balance sheet text:
        {text}
        
        JSON output:
        """
        
        try:
            response = self.query_ollama(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'(\{[\s\S]*\})', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response
                    
            try:
                result = json.loads(json_str)
                logger.info("Successfully parsed comparative data")
                return result
            except json.JSONDecodeError:
                logger.warning("Could not parse comparative data")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting comparative data: {str(e)}")
            return {}
    
    def convert_to_dataframe(self, data: Dict) -> pd.DataFrame:
        """
        Convert extracted balance sheet data to pandas DataFrame.
        
        Args:
            data: Dictionary with balance sheet data
            
        Returns:
            DataFrame representation of the balance sheet
        """
        rows = []
        
        def flatten_dict(d, prefix=''):
            items = []
            for k, v in d.items():
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, f"{prefix}{k}_"))
                else:
                    items.append((f"{prefix}{k}", v))
            return items
        
        # Check if we have comparative data
        if 'periods' in data:
            for period, period_data in data['periods'].items():
                flat_data = dict(flatten_dict(period_data))
                flat_data['period'] = period
                rows.append(flat_data)
        else:
            # Remove non-financial information for DataFrame
            financial_data = {k: v for k, v in data.items() if k not in ['company_name', 'statement_date']}
            rows.append(dict(flatten_dict(financial_data)))
            
        df = pd.DataFrame(rows)
        
        # Add metadata as attributes
        if 'company_name' in data:
            df.attrs['company_name'] = data['company_name']
        if 'statement_date' in data:
            df.attrs['statement_date'] = data['statement_date']
            
        return df
    
    def extract_balance_sheet(self, pdf_path: str, output_file: Optional[str] = None) -> Tuple[Dict, pd.DataFrame]:
        """
        Full extraction pipeline.
        
        Args:
            pdf_path: Path to the balance sheet PDF
            output_file: Optional path to save results
            
        Returns:
            Tuple of (raw data dict, pandas DataFrame)
        """
        logger.info(f"Starting extraction for: {pdf_path}")
        
        # Extract and preprocess text
        raw_text = self.extract_text_from_pdf(pdf_path)
        preprocessed_text = self.preprocess_text(raw_text)
        
        # Extract basic structure
        data = self.extract_balance_sheet_structure(preprocessed_text)
        
        # Try to extract comparative data
        comp_data = self.extract_comparative_data(preprocessed_text)
        if comp_data and 'periods' in comp_data:
            data['periods'] = comp_data['periods']
        
        # Convert to DataFrame
        df = self.convert_to_dataframe(data)
        
        # Save results if requested
        if output_file:
            file_base, file_ext = os.path.splitext(output_file)
            
            # Save raw JSON
            with open(f"{file_base}.json", 'w') as f:
                json.dump(data, f, indent=2)
                
            # Save CSV
            df.to_csv(f"{file_base}.csv", index=False)
            
            # Save Excel
            df.to_excel(f"{file_base}.xlsx", index=False)
            
            logger.info(f"Results saved to {file_base}.[json/csv/xlsx]")
        
        return data, df

def main():
    parser = argparse.ArgumentParser(description='Extract data from balance sheet PDFs with LLM assistance')
    parser.add_argument('pdf_path', help='Path to the balance sheet PDF')
    parser.add_argument('--output', '-o', help='Base output file path (without extension)')
    parser.add_argument('--ollama-url', default='http://localhost:11434/api/generate', help='Ollama API URL')
    parser.add_argument('--model', default='gemma3:latest', help='Ollama model to use')
    
    args = parser.parse_args()
    
    extractor = BalanceSheetExtractor(ollama_url=args.ollama_url)
    extractor.model = args.model
    
    try:
        data, df = extractor.extract_balance_sheet(args.pdf_path, args.output)
        
        # Print summary of extracted data
        print("\n=== Balance Sheet Extraction Summary ===")
        if 'company_name' in data:
            print(f"Company: {data['company_name']}")
        if 'statement_date' in data:
            print(f"Date: {data['statement_date']}")
            
        if 'total_assets' in data.get('assets', {}):
            print(f"Total Assets: {data['assets']['total_assets']}")
        if 'total_liabilities' in data.get('liabilities', {}):
            print(f"Total Liabilities: {data['liabilities']['total_liabilities']}")
        if 'total_equity' in data.get('equity', {}):
            print(f"Total Equity: {data['equity']['total_equity']}")
            
        print("\nDataFrame Shape:", df.shape)
        print("\nDataFrame Preview:")
        print(df.head())
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        parser.exit(1)

if __name__ == "__main__":
    main()