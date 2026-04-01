"""
Test script for the Credit Risk Assessment System
"""
import json
from src.app.ml.features import extract_features_from_transactions, features_to_dict
from src.app.ml.model import CreditRiskModel

# Sample transaction data (simulated bank statement)
sample_transactions = [
    {"date": "01/01/24", "description": "SALARY CREDIT", "withdrawal": None, "deposit": 50000.00, "balance": 75000.00},
    {"date": "02/01/24", "description": "RENT PAYMENT", "withdrawal": 15000.00, "deposit": None, "balance": 60000.00},
    {"date": "05/01/24", "description": "GROCERY STORE", "withdrawal": 3500.00, "deposit": None, "balance": 56500.00},
    {"date": "10/01/24", "description": "UTILITIES BILL", "withdrawal": 2500.00, "deposit": None, "balance": 54000.00},
    {"date": "15/01/24", "description": "FREELANCE PAYMENT", "withdrawal": None, "deposit": 15000.00, "balance": 69000.00},
    {"date": "20/01/24", "description": "CAR INSURANCE", "withdrawal": 5000.00, "deposit": None, "balance": 64000.00},
    {"date": "25/01/24", "description": "RESTAURANT", "withdrawal": 2000.00, "deposit": None, "balance": 62000.00},
    {"date": "28/01/24", "description": "ONLINE TRANSFER", "withdrawal": 3000.00, "deposit": None, "balance": 59000.00},
    {"date": "01/02/24", "description": "SALARY CREDIT", "withdrawal": None, "deposit": 50000.00, "balance": 109000.00},
    {"date": "02/02/24", "description": "RENT PAYMENT", "withdrawal": 15000.00, "deposit": None, "balance": 94000.00},
    {"date": "05/02/24", "description": "GROCERY STORE", "withdrawal": 4000.00, "deposit": None, "balance": 90000.00},
    {"date": "10/02/24", "description": "UTILITIES BILL", "withdrawal": 2300.00, "deposit": None, "balance": 87700.00},
    {"date": "15/02/24", "description": "INVESTMENT DIVIDEND", "withdrawal": None, "deposit": 5000.00, "balance": 92700.00},
    {"date": "20/02/24", "description": "GYM MEMBERSHIP", "withdrawal": 1500.00, "deposit": None, "balance": 91200.00},
    {"date": "25/02/24", "description": "SHOPPING", "withdrawal": 8000.00, "deposit": None, "balance": 83200.00},
    {"date": "01/03/24", "description": "SALARY CREDIT", "withdrawal": None, "deposit": 52000.00, "balance": 135200.00},
    {"date": "02/03/24", "description": "RENT PAYMENT", "withdrawal": 15000.00, "deposit": None, "balance": 120200.00},
    {"date": "05/03/24", "description": "GROCERY STORE", "withdrawal": 3800.00, "deposit": None, "balance": 116400.00},
    {"date": "10/03/24", "description": "UTILITIES BILL", "withdrawal": 2400.00, "deposit": None, "balance": 114000.00},
    {"date": "15/03/24", "description": "BONUS PAYMENT", "withdrawal": None, "deposit": 20000.00, "balance": 134000.00},
]

def test_feature_extraction():
    """Test feature extraction from transactions."""
    print("=" * 60)
    print("TEST 1: Feature Extraction")
    print("=" * 60)

    features = extract_features_from_transactions(sample_transactions)
    feature_dict = features_to_dict(features)

    print("\n📊 Extracted Features:")
    print(f"  Total Income (3 months): ${features.total_income_3m:,.2f}")
    print(f"  Avg Monthly Income: ${features.avg_monthly_income:,.2f}")
    print(f"  Income Stability: {features.income_stability:.2%}")
    print(f"  Income Sources: {features.income_sources}")
    print(f"  Total Expenses (3 months): ${features.total_expenses_3m:,.2f}")
    print(f"  Avg Balance: ${features.avg_balance:,.2f}")
    print(f"  Min Balance: ${features.min_balance:,.2f}")
    print(f"  Savings Rate: {features.savings_rate:.2%}")
    print(f"  Debt-to-Income: {features.debt_to_income:.2%}")
    print(f"  Creditworthiness Score: {features.creditworthiness_score:.1f}/100")

    return features

def test_credit_model(features):
    """Test the credit risk model."""
    print("\n" + "=" * 60)
    print("TEST 2: Credit Risk Prediction")
    print("=" * 60)

    model = CreditRiskModel()
    prediction = model.predict(features)

    print(f"\n📈 Loan Decision:")
    print(f"  Approved: {'✅ YES' if prediction['approved'] else '❌ NO'}")
    print(f"  Probability: {prediction['probability']:.2%}")
    print(f"  Credit Score: {prediction['credit_score']:.1f}/100")
    print(f"  Risk Level: {prediction['risk_level'].upper()}")

    print(f"\n📝 Decision Reasoning:")
    for reason in prediction['reasoning']:
        print(f"  • {reason}")

    if 'rule_score' in prediction:
        print(f"\n📊 Rule-based Score: {prediction['rule_score']}/100")

    return prediction

def test_high_risk_scenario():
    """Test with a high-risk applicant."""
    print("\n" + "=" * 60)
    print("TEST 3: High-Risk Applicant Scenario")
    print("=" * 60)

    # High-risk transactions (unstable income, frequent overdrafts)
    high_risk_transactions = [
        {"date": "01/01/24", "description": "FREELANCE PAYMENT", "withdrawal": None, "deposit": 20000.00, "balance": 5000.00},
        {"date": "02/01/24", "description": "RENT PAYMENT", "withdrawal": 12000.00, "deposit": None, "balance": -7000.00},  # Overdraft
        {"date": "05/01/24", "description": "OVERDRAFT FEE", "withdrawal": 500.00, "deposit": None, "balance": -7500.00},
        {"date": "10/01/24", "description": "CASH DEPOSIT", "withdrawal": None, "deposit": 10000.00, "balance": 2500.00},
        {"date": "15/01/24", "description": "LOAN REPAYMENT", "withdrawal": 8000.00, "deposit": None, "balance": -5500.00},  # Overdraft
        {"date": "20/01/24", "description": "PAYDAY LOAN", "withdrawal": None, "deposit": 5000.00, "balance": -500.00},
        {"date": "25/01/24", "description": "CREDIT CARD BILL", "withdrawal": 15000.00, "deposit": None, "balance": -15500.00},  # Overdraft
        {"date": "01/02/24", "description": "FREELANCE PAYMENT", "withdrawal": None, "deposit": 18000.00, "balance": 2500.00},
        {"date": "02/02/24", "description": "RENT PAYMENT", "withdrawal": 12000.00, "deposit": None, "balance": -9500.00},
        {"date": "10/02/24", "description": "CASH DEPOSIT", "withdrawal": None, "deposit": 12000.00, "balance": 2500.00},
        {"date": "15/02/24", "description": "LOAN REPAYMENT", "withdrawal": 8000.00, "deposit": None, "balance": -5500.00},
        {"date": "01/03/24", "description": "FREELANCE PAYMENT", "withdrawal": None, "deposit": 15000.00, "balance": 9500.00},
        {"date": "02/03/24", "description": "RENT PAYMENT", "withdrawal": 12000.00, "deposit": None, "balance": -2500.00},
    ]

    features = extract_features_from_transactions(high_risk_transactions)
    model = CreditRiskModel()
    prediction = model.predict(features)

    print(f"\n📈 Loan Decision:")
    print(f"  Approved: {'✅ YES' if prediction['approved'] else '❌ NO'}")
    print(f"  Probability: {prediction['probability']:.2%}")
    print(f"  Credit Score: {prediction['credit_score']:.1f}/100")
    print(f"  Risk Level: {prediction['risk_level'].upper()}")

    print(f"\n📝 Decision Reasoning:")
    for reason in prediction['reasoning']:
        print(f"  • {reason}")

    if 'red_flags' in prediction and prediction['red_flags']:
        print(f"\n🚩 Red Flags Detected:")
        for flag in prediction['red_flags']:
            print(f"  ⚠️  {flag}")

    return prediction

def main():
    print("\n" + "=" * 60)
    print("CREDIT RISK ASSESSMENT SYSTEM - TEST SUITE")
    print("=" * 60)

    # Run tests
    features = test_feature_extraction()
    prediction = test_credit_model(features)
    test_high_risk_scenario()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED ✅")
    print("=" * 60)

if __name__ == "__main__":
    main()
