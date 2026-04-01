import re
from pdfminer.high_level import extract_text

text = extract_text("ACC_Sample.pdf")
lines = text.split("\n")

date_pattern = re.compile(r"\d{2}/\d{2}/\d{2}")
amount_pattern = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$")

transactions = []
current_txn = None
amounts_collected = []
previous_balance = None

for line in lines:
    line = line.strip()

    # Start of transaction: date + narration
    if re.match(r"\d{2}/\d{2}/\d{2}\s+\S+", line):
        if current_txn:
            transactions.append(current_txn)

        current_txn = {
            "date": line[:8],
            "description": line[9:].strip(),
            "withdrawal": None,
            "deposit": None,
            "balance": None,
        }
        amounts_collected = []

    elif current_txn:
        # Standalone amount line
        if amount_pattern.match(line):
            amounts_collected.append(line)

            if len(amounts_collected) == 2:
                amount = float(amounts_collected[0].replace(",", ""))
                balance = float(amounts_collected[1].replace(",", ""))

                current_txn["balance"] = balance

                if previous_balance is not None:
                    if balance < previous_balance:
                        current_txn["withdrawal"] = amount
                    else:
                        current_txn["deposit"] = amount
                else:
                    # First transaction fallback
                    current_txn["withdrawal"] = amount

                previous_balance = balance

        # Ignore standalone value date lines
        elif date_pattern.fullmatch(line):
            continue

        else:
            current_txn["description"] += " " + line

if current_txn:
    transactions.append(current_txn)

print(*transactions[:10], sep="\n")
