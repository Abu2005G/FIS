"""
Credit Risk Assessment ML Model
Uses Random Forest classifier with engineered financial features
"""
import pickle
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import numpy as np
from pathlib import Path


try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


from src.app.ml.features import FinancialFeatures, features_to_dict


class CreditRiskModel:
    """
    Credit Risk Assessment Model

    Predicts loan approval probability based on financial features.
    Uses a rules-based fallback if scikit-learn is not available.
    """

    # Feature names in order for model input
    FEATURE_NAMES = [
        'avg_monthly_income',
        'income_stability',
        'income_sources',
        'total_expenses_3m',
        'essential_expense_ratio',
        'avg_balance',
        'min_balance',
        'balance_trend',
        'negative_balance_count',
        'transaction_count',
        'large_transactions',
        'debt_to_income',
        'savings_rate',
        'expense_to_income',
        'creditworthiness_score'
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.model_path = model_path or self._get_default_model_path()
        self._load_or_initialize()

    def _get_default_model_path(self) -> str:
        """Get default path for model storage."""
        base_dir = Path(__file__).parent
        return str(base_dir / "credit_risk_model.pkl")

    def _load_or_initialize(self):
        """Load existing model or initialize rules-based fallback."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.scaler = data.get('scaler')
            except Exception as e:
                print(f"Could not load model: {e}")
                self.model = None

    def _extract_feature_vector(self, features: FinancialFeatures) -> np.ndarray:
        """Extract feature vector from FinancialFeatures."""
        feature_dict = features_to_dict(features)
        return np.array([feature_dict.get(name, 0) for name in self.FEATURE_NAMES])

    def predict(self, features: FinancialFeatures) -> Dict[str, Any]:
        """
        Predict credit risk and make loan decision.

        Returns:
            Dictionary with prediction results including:
            - approved: bool
            - probability: float (0-1)
            - credit_score: float (0-100)
            - risk_level: str ('low', 'medium', 'high')
            - reasoning: list of reasons for decision
        """
        if self.model and SKLEARN_AVAILABLE:
            return self._ml_predict(features)
        else:
            return self._rules_based_predict(features)

    def _ml_predict(self, features: FinancialFeatures) -> Dict[str, Any]:
        """ML-based prediction using trained model."""
        feature_vector = self._extract_feature_vector(features)
        feature_vector = feature_vector.reshape(1, -1)

        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)

        probability = self.model.predict_proba(feature_vector)[0][1]  # Probability of approval
        approved = probability > 0.5

        # Determine risk level
        if probability >= 0.8:
            risk_level = "low"
        elif probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Generate reasoning
        reasoning = self._generate_reasoning(features, approved)

        return {
            "approved": approved,
            "probability": round(probability, 4),
            "credit_score": features.creditworthiness_score,
            "risk_level": risk_level,
            "reasoning": reasoning
        }

    def _rules_based_predict(self, features: FinancialFeatures) -> Dict[str, Any]:
        """
        Rules-based prediction when ML model is not available.
        Uses weighted scoring and decision rules.
        """
        score = 0
        max_score = 100
        reasoning = []
        red_flags = []

        # Income stability (up to 20 points)
        if features.income_stability >= 0.7:
            score += 20
            reasoning.append("Stable income source verified")
        elif features.income_stability >= 0.4:
            score += 10
            reasoning.append("Moderate income stability")
        else:
            red_flags.append("Unstable income pattern detected")

        # Savings rate (up to 15 points)
        if features.savings_rate >= 0.2:
            score += 15
            reasoning.append(f"Strong savings rate: {features.savings_rate*100:.1f}%")
        elif features.savings_rate >= 0.05:
            score += 8
        else:
            red_flags.append("Low savings rate indicates financial stress")

        # Debt-to-income ratio (up to 15 points, inverted)
        if features.debt_to_income <= 0.1:
            score += 15
            reasoning.append("Low debt burden")
        elif features.debt_to_income <= 0.3:
            score += 8
        else:
            score -= 10
            red_flags.append(f"High debt-to-income ratio: {features.debt_to_income*100:.1f}%")

        # Balance management (up to 15 points)
        if features.avg_balance > features.avg_monthly_income * 0.5:
            score += 15
            reasoning.append("Healthy average balance maintained")
        elif features.avg_balance > 0:
            score += 5
        else:
            red_flags.append("Negative average balance")

        # No overdrafts (up to 10 points)
        if features.negative_balance_count == 0:
            score += 10
            reasoning.append("No overdraft incidents")
        else:
            score -= min(features.negative_balance_count * 5, 15)
            red_flags.append(f"{features.negative_balance_count} overdraft(s) detected")

        # Expense ratio (up to 10 points)
        if features.expense_to_income <= 0.7:
            score += 10
        elif features.expense_to_income <= 0.9:
            score += 5
        else:
            red_flags.append("Expenses exceed 90% of income")

        # Transaction patterns (up to 5 points)
        if features.large_transactions <= 2:
            score += 5
        if features.frequent_small_deposits > 5:
            red_flags.append("Unusual deposit pattern detected")
            score -= 5

        # Balance trend (up to 10 points)
        if features.balance_trend > 0:
            score += 10
            reasoning.append("Positive balance trend")
        elif features.balance_trend < -1000:
            red_flags.append("Declining balance trend")

        # Final scoring
        final_score = max(0, min(score, max_score))
        probability = final_score / max_score

        # Decision logic
        if final_score >= 60 and not red_flags:
            approved = True
            risk_level = "low"
        elif final_score >= 40 and len(red_flags) <= 1:
            approved = True
            risk_level = "medium"
        else:
            approved = False
            risk_level = "high"
            reasoning.extend(red_flags)

        # Add score-based reasoning
        if final_score >= 70:
            reasoning.insert(0, "Strong overall financial profile")
        elif final_score >= 50:
            reasoning.insert(0, "Moderate financial profile")
        else:
            reasoning.insert(0, "Financial profile needs improvement")

        return {
            "approved": approved,
            "probability": round(probability, 4),
            "credit_score": features.creditworthiness_score,
            "risk_level": risk_level,
            "reasoning": reasoning,
            "rule_score": final_score,
            "red_flags": red_flags
        }

    def _generate_reasoning(self, features: FinancialFeatures, approved: bool) -> List[str]:
        """Generate human-readable reasoning for the decision."""
        reasoning = []

        if features.income_stability > 0.7:
            reasoning.append("Income is stable and predictable")

        if features.savings_rate > 0.15:
            reasoning.append("Good savings discipline demonstrated")

        if features.debt_to_income < 0.2:
            reasoning.append("Low debt obligations")

        if features.avg_balance > features.avg_monthly_income:
            reasoning.append("Maintains healthy cash reserves")

        if features.negative_balance_count == 0:
            reasoning.append("No account overdrafts")
        elif features.negative_balance_count > 2:
            reasoning.append("Multiple overdraft incidents")

        if not approved:
            if features.creditworthiness_score < 40:
                reasoning.append("Overall creditworthiness score too low")
            if features.expense_to_income > 0.9:
                reasoning.append("High expense-to-income ratio")
            if features.negative_balance_count > 3:
                reasoning.append("Frequent overdrafts indicate financial stress")

        return reasoning

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on historical data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=reject, 1=approve)
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")

        # Save model
        self.save()

        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score
        }

    def save(self):
        """Save model to disk."""
        data = {
            'model': self.model,
            'scaler': self.scaler
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.model or not hasattr(self.model, 'feature_importances_'):
            return {}

        importance = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.FEATURE_NAMES, importance)
        }


# Global model instance
_credit_model = None

def get_credit_model() -> CreditRiskModel:
    """Get or create the global credit model instance."""
    global _credit_model
    if _credit_model is None:
        _credit_model = CreditRiskModel()
    return _credit_model
