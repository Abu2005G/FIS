"""
PDF Statement Parser for Financial Document Processing
Integrates with the existing transaction_parser to extract structured data
"""
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


def parse_statement_pdf(file_path: str) -> Dict[str, Any]:
    """
    Parse a bank statement PDF and extract transactions.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary with extracted transactions and metadata
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Try different parsers
    if PDFPLUMBER_AVAILABLE:
        return _parse_with_pdfplumber(file_path)
    elif PDFMINER_AVAILABLE:
        return _parse_with_pdfminer(file_path)
    else:
        raise RuntimeError("No PDF parsing library available. Install pdfplumber or pdfminer.six")


def _parse_with_pdfplumber(file_path: str) -> Dict[str, Any]:
    """Parse PDF using pdfplumber (better for tables)."""
    transactions = []
    metadata = {}

    with pdfplumber.open(file_path) as pdf:
        # Extract metadata from first page
        first_page = pdf.pages[0]
        text = first_page.extract_text()

        # Try to extract account info
        metadata = _extract_metadata(text)

        # Extract transactions from all pages
        for page in pdf.pages:
            page_text = page.extract_text()
            page_transactions = _extract_transactions_from_text(page_text)
            transactions.extend(page_transactions)

    return {
        "metadata": metadata,
        "transactions": transactions,
        "transaction_count": len(transactions)
    }


def _parse_with_pdfminer(file_path: str) -> Dict[str, Any]:
    """Parse PDF using pdfminer.six."""
    text = extract_text(file_path)
    transactions = _extract_transactions_from_text(text)

    # Extract metadata from first 1000 characters
    metadata = _extract_metadata(text[:1000])

    return {
        "metadata": metadata,
        "transactions": transactions,
        "transaction_count": len(transactions)
    }


def _extract_metadata(text: str) -> Dict[str, Any]:
    """Extract account metadata from statement."""
    metadata = {}

    # Account number patterns
    account_patterns = [
        r'Account\s*(?:Number|No|#)[:\s]*([\d\-X]+)',
        r'Account[:\s]*([\d\-X]+)',
        r'Acct\s*(?:#|Number)[:\s]*([\d\-X]+)'
    ]

    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['account_number'] = match.group(1).strip()
            break

    # Statement period
    period_patterns = [
        r'(?:Statement|Period)\s*(?:Period|Date|From)[:\s]*(\w+\s+\d{1,2},?\s*\d{4}.*?\d{1,2},?\s*\d{4})',
        r'From[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}).*?To[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
    ]

    for pattern in period_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['statement_period'] = match.group(0)
            break

    # Bank name
    bank_patterns = [
        r'^([A-Z][A-Za-z\s]+Bank)',
        r'([A-Z][A-Za-z\s]+Bank\s+(?:Limited|Ltd|Inc|Corp))'
    ]

    for pattern in bank_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            metadata['bank_name'] = match.group(1).strip()
            break

    return metadata


def _extract_transactions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract transactions from statement text.
    Supports various statement formats.
    """
    transactions = []
    lines = text.split('\n')

    # Common date patterns
    date_pattern = re.compile(r'^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})')

    # Amount patterns
    amount_pattern = re.compile(r'[\d,]+\.\d{2}')

    current_transaction = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a date
        date_match = date_pattern.match(line)

        if date_match:
            # Save previous transaction if exists
            if current_transaction:
                transactions.append(current_transaction)

            # Start new transaction
            date_str = date_match.group(1)
            rest_of_line = line[len(date_str):].strip()

            current_transaction = {
                "date": date_str,
                "description": "",
                "withdrawal": None,
                "deposit": None,
                "balance": None
            }

            # Extract amounts from the rest of the line
            amounts = amount_pattern.findall(rest_of_line)

            if amounts:
                # Remove amounts from description
                for amount in amounts:
                    rest_of_line = rest_of_line.replace(amount, "")
                rest_of_line = rest_of_line.strip()

                # Parse amounts
                parsed_amounts = [float(a.replace(',', '')) for a in amounts]

                # Assign based on position (withdrawal, deposit, balance)
                if len(parsed_amounts) >= 2:
                    current_transaction["balance"] = parsed_amounts[-1]
                    # Determine if withdrawal or deposit based on context
                    if len(parsed_amounts) >= 3:
                        current_transaction["withdrawal"] = parsed_amounts[-3] if parsed_amounts[-3] > 0 else None
                        current_transaction["deposit"] = parsed_amounts[-2] if parsed_amounts[-2] > 0 else None
                    else:
                        # Single amount - need to infer from description or balance change
                        current_transaction["withdrawal"] = parsed_amounts[-2] if parsed_amounts[-2] > 0 else None

            current_transaction["description"] = rest_of_line

        elif current_transaction:
            # Continue previous transaction description
            amounts = amount_pattern.findall(line)
            if amounts:
                for amount in amounts:
                    line = line.replace(amount, "")
                    parsed = float(amount.replace(',', ''))

                    if current_transaction.get("balance") is None:
                        current_transaction["balance"] = parsed
                    elif current_transaction.get("withdrawal") is None:
                        current_transaction["withdrawal"] = parsed
                    elif current_transaction.get("deposit") is None:
                        current_transaction["deposit"] = parsed

            current_transaction["description"] += " " + line.strip()

    # Don't forget the last transaction
    if current_transaction:
        transactions.append(current_transaction)

    # Clean up and filter transactions
    cleaned = []
    for txn in transactions:
        if txn.get("date") and txn.get("description"):
            # Clean up description
            txn["description"] = re.sub(r'\s+', ' ', txn["description"]).strip()
            cleaned.append(txn)

    return cleaned


def parse_transaction_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Main entry point for parsing transaction files.
    Supports PDF and JSON formats.

    Args:
        file_path: Path to the file

    Returns:
        List of transaction dictionaries
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == '.pdf':
        result = parse_statement_pdf(file_path)
        return result.get("transactions", [])
    elif suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("transactions", [])
    elif suffix in ['.csv', '.txt']:
        # Add CSV parsing if needed
        raise NotImplementedError(f"File format not yet supported: {suffix}")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


if __name__ == "__main__":
    # Test parser
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            result = parse_statement_pdf(file_path)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python statement_parser.py <pdf_path>")
