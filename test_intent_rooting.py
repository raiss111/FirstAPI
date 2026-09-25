import unittest

from intent_agent import (
    FALLBACK_INTENT,
    INTENTS,
    RECOMMENDATION_INTENTS,
    TOBI_INTENTS,
    get_intent_scope,
)


class IntentRoutingTests(unittest.TestCase):
    def test_tobi_intents_are_present(self):
        self.assertEqual(
            set(TOBI_INTENTS),
            {
                "explain_tariff_plan",
                "topup_airtime",
                "check_airtime_balance",
                "handover_human",
            },
        )

    def test_old_banking_intents_are_removed(self):
        old_banking = {
            "credit_request",
            "credit_information",
            "credit_requirements",
            "credit_repayment",
            "account_information",
            "account_balance",
            "transaction_information",
        }
        self.assertTrue(old_banking.isdisjoint(set(INTENTS)))

    def test_recommendation_intents_are_unchanged(self):
        self.assertEqual(
            RECOMMENDATION_INTENTS,
            (
                "BUY_PREPAID",
                "ASK_RECOMMENDATION",
                "RECOMMEND_DEVICE",
                "SUMMARIZE",
            ),
        )

    def test_scopes(self):
        for intent in TOBI_INTENTS:
            self.assertEqual(get_intent_scope(intent), "tobi")

        for intent in RECOMMENDATION_INTENTS:
            self.assertEqual(get_intent_scope(intent), "recommendation")

        self.assertEqual(get_intent_scope(FALLBACK_INTENT), "unknown")


if __name__ == "__main__":
    unittest.main()
