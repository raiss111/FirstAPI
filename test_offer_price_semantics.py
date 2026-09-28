"""Régressions MVP: montants exacts, plafonds, aucune offre, réponse sûre.

Dans le venv du projet :
    python -m unittest test_offer_price_semantics -v
Aucun appel réseau réel ni achat réel n'est nécessaire.
"""
import importlib.util
import json
import os
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Substituts facultatifs pour permettre l'exécution hors du venv du projet.
if importlib.util.find_spec('groq') is None:
    sdk = types.ModuleType('groq')
    sdk.__spec__ = ModuleSpec('groq', loader=None)
    sdk.Groq = type('Groq', (), {'__init__': lambda self, **kwargs: None})
    sdk.APIError = type('APIError', (Exception,), {})
    sys.modules['groq'] = sdk
if importlib.util.find_spec('langchain_core') is None:
    core = types.ModuleType('langchain_core')
    core.__path__ = []
    core.__spec__ = ModuleSpec('langchain_core', loader=None, is_package=True)
    mod = types.ModuleType('langchain_core.tools')
    class _TestTool:
        def __init__(self, func):
            self.func = func
        def invoke(self, inputs):
            return self.func(**inputs)
    mod.tool = lambda fn: _TestTool(fn)
    sys.modules['langchain_core'] = core
    sys.modules['langchain_core.tools'] = mod
if importlib.util.find_spec('langgraph') is None:
    import nodes as _nodes
    graph = types.ModuleType('recommendation_graph')
    def _invoke(inputs):
        state = dict(inputs)
        state.update(_nodes.extract_context_node(state))
        branch = {
            'BUY_PREPAID': _nodes.get_offers_node,
            'ASK_RECOMMENDATION': _nodes.recommend_offer_node,
            'RECOMMEND_DEVICE': _nodes.get_devices_node,
            'SUMMARIZE': _nodes.summary_node,
        }.get(state.get('intent'))
        if branch:
            state.update(branch(state))
        return state
    graph.recommendation_graph = SimpleNamespace(invoke=_invoke)
    sys.modules['recommendation_graph'] = graph

import nodes
import communication_agent as ca
from llm_response_generator import (
    _no_offer_safe_fallback, _validate_recommendation_response, generate_user_response,
)


def _reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _offer_turn(message, previous=None):
    state = {**(previous or {}), 'intent': 'BUY_PREPAID', 'message': message}
    state.update(nodes.extract_context_node(state))
    state.update(nodes.get_offers_node(state))
    return state


def _unavailable_context(state):
    return {
        'current_request': {'intent': 'BUY_PREPAID', 'scope': 'recommendation'},
        'recommendation_state': state,
        'selected_offer': None,
        'purchase_state': {'status': 'OFFER_UNAVAILABLE', 'offer': None},
    }


class PriceSemanticsTests(unittest.TestCase):
    def test_two_dollars_is_normal_no_match(self):
        state = _offer_turn('je veux acheter un forfait de 2$')
        self.assertEqual(state['price_match_mode'], 'exact')
        self.assertEqual(state['offers'], [])
        self.assertIsNone(state['selected_offer_id'])
        self.assertEqual(state['lowest_available_offer']['price_monthly'], 8)

    def test_forty_and_fifty_are_exact_requests_not_a_purchase_of_fifteen(self):
        for amount in (40, 50):
            with self.subTest(amount=amount):
                state = _offer_turn(f'je veux acheter un forfait de {amount}$')
                self.assertEqual(state['price_match_mode'], 'exact')
                self.assertEqual(state['budget'], amount)
                self.assertEqual(state['offers'], [])
                self.assertEqual(state['selected_offer_id'], None)
                self.assertEqual({o['price_monthly'] for o in state['available_offers']}, {8, 15})

    def test_explicit_budget_ceiling_remains_a_ceiling(self):
        state = _offer_turn('je veux acheter un forfait, mon budget maximum est 40$')
        self.assertEqual(state['price_match_mode'], 'maximum')
        self.assertEqual({o['price_monthly'] for o in state['offers']}, {8, 15})
        self.assertEqual(state['selected_offer_id'], 'PRE_001')

    def test_matching_prices_eight_and_fifteen_remain_unchanged(self):
        for amount, offer_id in ((8, 'PRE_002'), (15, 'PRE_001')):
            with self.subTest(amount=amount):
                state = _offer_turn(f'achète un forfait de {amount}$')
                self.assertEqual(state['selected_offer_id'], offer_id)
                self.assertEqual([o['offer_id'] for o in state['offers']], [offer_id])

    def test_no_reuse_of_previous_offer_for_new_exact_request(self):
        old = _offer_turn('achète un forfait de 8$ pour Ruben')
        current = _offer_turn('je veux acheter un forfait de 50$', old)
        self.assertIsNone(current['selected_offer_id'])
        self.assertEqual(current['offers'], [])
        self.assertEqual(current['price_match_mode'], 'exact')

    def test_plain_number_without_currency_does_not_gain_usd(self):
        old = _offer_turn('achète un forfait de 8$')
        state = _offer_turn('mon budget est 40', old)
        self.assertEqual(state['price_match_mode'], 'maximum')
        self.assertIsNone(state['budget_currency'])


class ResponseNoOfferTests(unittest.TestCase):
    def test_fifty_and_catalogue_alternatives_are_allowed_facts(self):
        state = _offer_turn('je veux acheter un forfait de 50$')
        answer = "Aucun forfait au prix exact de 50 USD. Eco Prepaid coûte 8 USD et Flexi Data Max coûte 15 USD."
        self.assertEqual(_validate_recommendation_response(answer, _unavailable_context(state)), [])

    def test_wrong_euro_and_invented_offer_still_rejected(self):
        state = _offer_turn('je veux un forfait de 40$')
        context = _unavailable_context(state)
        self.assertIn('unsupported_currency:EUR', _validate_recommendation_response('Pas de forfait à 40 EUR.', context))
        self.assertIn('unsupported_price_currency_pair', _validate_recommendation_response('Un forfait coûte 30 USD.', context))

    def test_no_offer_must_never_ask_for_confirmation(self):
        context = _unavailable_context(_offer_turn('je veux acheter un forfait de 50$'))
        issues = _validate_recommendation_response('Aucun forfait à 50 USD. Confirmez-vous l’achat simulé ?', context)
        self.assertIn('unavailable_offer_confirmation_not_allowed', issues)

    def test_after_two_bad_llm_answers_a_normal_no_match_does_not_become_http_502(self):
        state = _offer_turn('je veux acheter un forfait de 50$')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(
            side_effect=[_reply('Confirmez-vous votre achat de 15 euros ?'),
                         _reply('Achetez maintenant sur une application !')],
        ))))
        answer = generate_user_response(
            client, user_message='je veux acheter un forfait de 50$',
            business_context=_unavailable_context(state), history=[], preferred_language='fr',
        )
        self.assertEqual(answer, _no_offer_safe_fallback(_unavailable_context(state)))
        self.assertIn('50 USD', answer)
        self.assertIn('8 USD', answer)
        self.assertIn('15 USD', answer)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        for decoration in ('\n', '\\n', '**', '||', '____'):
            self.assertNotIn(decoration, answer)


class ConversationContinuityTests(unittest.TestCase):
    def setUp(self):
        ca._CONVERSATION_STATE.clear()
        ca._CONVERSATION_STATE.update({
            'customer_id': None, 'profile': {}, 'history': [], 'last_intent': None,
            'current_intent': None, 'current_scope': 'unknown', 'recommendation': {},
            'tobi': {}, 'purchase': {},
        })

    def test_ruben_purchase_device_followed_by_fifty_never_confirms_fifteen(self):
        intent = [
            {'intent': 'BUY_PREPAID', 'purchase_action': 'request', 'beneficiary_name': 'Ruben'},
            {'intent': 'BUY_PREPAID', 'purchase_action': 'confirm', 'beneficiary_name': None},
            {'intent': 'RECOMMEND_DEVICE'},
            {'intent': 'BUY_PREPAID', 'purchase_action': 'request', 'beneficiary_name': None},
        ]
        replies = [
            'Confirmez-vous cet achat simulé de Eco Prepaid à 8 USD pour Ruben?',
            "L'achat simulé du forfait Eco Prepaid pour Ruben a été effectué avec succès.",
            'Je recommande le Xiaomi Redmi Note 13 à 150 USD.',
            "Aucun forfait au prix exact de 50 USD. Les offres disponibles coûtent 8 USD et 15 USD.",
        ]
        with patch.object(ca, 'classify_intent', side_effect=intent), \
             patch.object(ca, 'generate_user_response', side_effect=replies), \
             patch.object(ca, 'purchase_prepaid_mock') as purchase_mock:
            purchase_mock.invoke.return_value = {
                'status': 'SIMULATED_SUCCESS', 'mode': 'mock', 'real_transaction': False,
            }
            first = ca.handle_user_message(object(), customer_id='CUST_001', message='achète un forfait de 8$ pour Ruben')
            second = ca.handle_user_message(object(), message='oui je confirme')
            device = ca.handle_user_message(object(), message='quel smartphone pour Ruben ?')
            no_match = ca.handle_user_message(object(), message='je veux acheter un forfait de 50$')
        self.assertEqual(first['data']['recommendation']['selected_offer_id'], 'PRE_002')
        self.assertEqual(second['data']['purchase']['status'], 'COMPLETED')
        self.assertEqual([d['device_id'] for d in device['data']['recommendation']['devices']], ['DEV_102'])
        self.assertIsNone(no_match['data']['recommendation']['selected_offer_id'])
        self.assertEqual(no_match['data']['purchase']['status'], 'OFFER_UNAVAILABLE')
        self.assertEqual(purchase_mock.invoke.call_count, 1)
        self.assertEqual(len(ca.get_conversation_state()['history']), 8)


class ChatHttpNoOfferTests(unittest.TestCase):
    def test_two_forty_fifty_requests_return_http_200_not_validation_errors(self):
        # Même conversation MVP ; le candidat LLM est volontairement faux pour
        # vérifier le repli factuel et le contrat réel de la route FastAPI.
        import importlib
        from fastapi.testclient import TestClient
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-only-placeholder'}):
            with patch('groq.Groq', return_value=object()):
                import main
                importlib.reload(main)
        ca._CONVERSATION_STATE.clear()
        ca._CONVERSATION_STATE.update({
            'customer_id': None, 'profile': {}, 'history': [], 'last_intent': None,
            'current_intent': None, 'current_scope': 'unknown', 'recommendation': {},
            'tobi': {}, 'purchase': {},
        })
        fake_model = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(return_value=_reply('Confirmez-vous cet achat simulé de 15 EUR ?')),
        )))
        main.client = fake_model
        with patch.object(ca, 'classify_intent', return_value={
            'intent': 'BUY_PREPAID', 'purchase_action': 'request', 'beneficiary_name': None,
        }):
            for amount in (2, 40, 50):
                response = TestClient(main.app).post('/chat', json={
                    'customer_id': 'CUST_001',
                    'message': f'je veux acheter un forfait de {amount}$',
                })
                with self.subTest(amount=amount):
                    self.assertEqual(response.status_code, 200, response.text)
                    body = response.json()
                    self.assertEqual(body['state']['purchase']['status'], 'OFFER_UNAVAILABLE')
                    self.assertIsNone(body['state']['selected_offer_id'])
                    self.assertEqual(body['offers'], [])
                    self.assertIn(f'{amount} USD', body['response'])
                    self.assertNotIn('\n', body['response'])
        self.assertEqual(fake_model.chat.completions.create.call_count, 6)
