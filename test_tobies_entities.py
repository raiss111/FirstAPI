import unittest

from intent_agent import extract_tobi_entities


class TobiEntityExtractionTests(unittest.TestCase):
    def test_topup_other_complete(self):
        result = extract_tobi_entities(
            "topup_airtime",
            "Acheter 5$ d'unités pour le 0812345678 via M-Pesa",
        )

        self.assertTrue(result["slots_complete"])
        self.assertEqual(result["entities"]["beneficiary_type"], "other")
        self.assertEqual(result["entities"]["target_msisdn"], "+243812345678")
        self.assertEqual(result["entities"]["amount"], 5)
        self.assertEqual(result["entities"]["currency"], "USD")
        self.assertEqual(result["entities"]["payment_method"], "mpesa")

    def test_topup_self_uses_mpesa_default(self):
        result = extract_tobi_entities(
            "topup_airtime",
            "Recharge-moi 10000 CDF",
        )

        self.assertTrue(result["slots_complete"])
        self.assertEqual(result["entities"]["beneficiary_type"], "self")
        self.assertEqual(result["entities"]["amount"], 10000)
        self.assertEqual(result["entities"]["currency"], "CDF")
        self.assertEqual(result["entities"]["payment_method"], "mpesa")
        self.assertNotIn("target_msisdn", result["entities"])

    def test_topup_missing_amount_is_incomplete(self):
        result = extract_tobi_entities(
            "topup_airtime",
            "Je veux recharger du crédit via M-Pesa",
        )

        self.assertFalse(result["slots_complete"])

    def test_handover_chat_complete(self):
        result = extract_tobi_entities(
            "handover_human",
            "Je veux parler à un conseiller par chat",
        )

        self.assertTrue(result["slots_complete"])
        self.assertEqual(result["entities"]["preferred_channel"], "chat")

    def test_handover_without_channel_is_incomplete(self):
        result = extract_tobi_entities(
            "handover_human",
            "Je veux parler à un conseiller",
        )

        self.assertFalse(result["slots_complete"])

    def test_check_balance_has_no_required_entity(self):
        result = extract_tobi_entities(
            "check_airtime_balance",
            "Quel est mon solde Airtime ?",
        )

        self.assertTrue(result["slots_complete"])
        self.assertEqual(result["entities"], {})

    def test_plan_name_is_optional(self):
        result = extract_tobi_entities(
            "explain_tariff_plan",
            "Explique-moi mon plan tarifaire actuel",
        )

        self.assertTrue(result["slots_complete"])

    def test_recommendation_intent_is_untouched(self):
        result = extract_tobi_entities(
            "BUY_PREPAID",
            "Je cherche un forfait prépayé à 20$",
        )

        self.assertEqual(result, {"entities": {}, "slots_complete": True})


if __name__ == "__main__":
    unittest.main()
