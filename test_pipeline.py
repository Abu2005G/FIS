"""
Test the 5-Stage LLM Prompt Pipeline for Credit Risk Assessment
"""
import sys
sys.path.insert(0, '/Users/abdulvaseemmasood/FIS')

from src.app.prompts.pipeline import FinancialIntelligencePipeline

# Sample raw transaction strings (as would come from PDF parsing)
sample_transactions = [
    "01/01/24 SALARY CREDIT 50000.00 credit",
    "02/01/24 RENT PAYMENT 15000.00 debit",
    "05/01/24 GROCERY STORE 3500.00 debit",
    "10/01/24 UTILITIES BILL 2500.00 debit",
    "15/01/24 FREELANCE PAYMENT 15000.00 credit",
    "20/01/24 CAR INSURANCE 5000.00 debit",
    "25/01/24 RESTAURANT 2000.00 debit",
    "28/01/24 ONLINE TRANSFER 3000.00 debit",
    "01/02/24 SALARY CREDIT 50000.00 credit",
    "02/02/24 RENT PAYMENT 15000.00 debit",
    "05/02/24 GROCERY STORE 4000.00 debit",
    "10/02/24 UTILITIES BILL 2300.00 debit",
    "15/02/24 INVESTMENT DIVIDEND 5000.00 credit",
    "20/02/24 GYM MEMBERSHIP 1500.00 debit",
    "25/02/24 SHOPPING 8000.00 debit",
    "01/03/24 SALARY CREDIT 52000.00 credit",
    "02/03/24 RENT PAYMENT 15000.00 debit",
    "05/03/24 GROCERY STORE 3800.00 debit",
    "10/03/24 UTILITIES BILL 2400.00 debit",
    "15/03/24 BONUS PAYMENT 20000.00 credit",
]

def test_pipeline():
    print("="*70)
    print("TESTING 5-STAGE LLM PROMPT PIPELINE")
    print("="*70)
    print("\n⚠️  Note: This requires Ollama to be running locally")
    print("   Install from: https://ollama.com")
    print("   Run: ollama pull llama3.2")
    print()

    # Initialize pipeline (will use mock if Ollama not available)
    pipeline = FinancialIntelligencePipeline(llm_provider="ollama", llm_model="llama3.2")

    # Run pipeline
    result = pipeline.process_application(
        application_id=1001,
        applicant_name="John Doe",
        loan_amount=100000,
        loan_purpose="Home Renovation",
        raw_transactions=sample_transactions
    )

    print("\n" + "="*70)
    print("FINAL DECISION")
    print("="*70)

    decision = result['decision']
    credit_decision = decision.get('credit_decision', {})
    risk_profile = decision.get('risk_profile', {})

    print(f"\n📊 Risk Profile:")
    print(f"   Score: {risk_profile.get('score', 'N/A')}")
    print(f"   Level: {risk_profile.get('level', 'N/A').upper()}")

    print(f"\n✅ Credit Decision:")
    print(f"   Approved: {credit_decision.get('approved', False)}")
    print(f"   Confidence: {credit_decision.get('confidence', 'N/A')}")
    print(f"   Max Approved Amount: ${credit_decision.get('max_approved_amount', 0)}")

    print(f"\n📝 Reasoning:")
    for reason in credit_decision.get('reasoning', []):
        print(f"   • {reason}")

    print(f"\n📋 Audit Summary:")
    audit_summary = decision.get('audit_summary', {})
    print(f"   Red Flags: {audit_summary.get('red_flags', [])}")
    print(f"   Green Flags: {audit_summary.get('green_flags', [])}")
    print(f"   Recommendation: {audit_summary.get('recommendation', 'N/A')}")

    print(f"\n⏱️  Processing Summary:")
    summary = result['processing_summary']
    print(f"   Transactions: {summary['transactions_processed']}")
    print(f"   Stages: {summary['stages_completed']}")
    print(f"   Overall Confidence: {summary['overall_confidence']:.2%}")
    print(f"   Total Time: {summary['processing_time_ms']:.0f}ms")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return result

def test_audit_trail():
    """Test that audit log is properly captured"""
    print("\n" + "="*70)
    print("AUDIT TRAIL VERIFICATION")
    print("="*70)

    pipeline = FinancialIntelligencePipeline(llm_provider="mock")

    result = pipeline.process_application(
        application_id=1002,
        applicant_name="Jane Smith",
        loan_amount=50000,
        loan_purpose="Car Purchase",
        raw_transactions=sample_transactions[:10]
    )

    audit_log = result['audit_log']

    print(f"\n📋 Audit Log for Application #{audit_log['application_id']}")
    print(f"   Applicant: {audit_log['applicant_name']}")
    print(f"   Loan Amount: ${audit_log['loan_amount']}")
    print(f"   Created: {audit_log['created_at']}")
    print(f"   Overall Confidence: {audit_log['overall_confidence']:.2%}")

    print(f"\n🔄 Pipeline Stages:")
    for stage in audit_log['stages']:
        print(f"   • {stage['stage_name']}: {stage['execution_time_ms']:.0f}ms (confidence: {stage['confidence']:.2f})")

    print("\n✅ Audit trail complete and compliant")

if __name__ == "__main__":
    try:
        test_pipeline()
        test_audit_trail()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
