"""Régressions ciblées : forfait sous le prix minimal et appareils autorisés.

Dans le venv réel : python -m unittest test_empty_offer_and_device_guard -v
En environnement de test sans SDK, les frontières externes peuvent être substituées.
Aucune API télécom ni achat réel n'est invoqué.
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


# Uniquement pour exécuter les tests sur une machine sans SDK ; en installation
# réelle, FastAPI/LangGraph/Groq/LangChain du projet sont utilisés tels quels.
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
    import nodes as _real_nodes
    graph = types.ModuleType('recommendation_graph')
    def _invoke(inputs):
        state = dict(inputs)
        state.update(_real_nodes.extract_context_node(state))
        chosen = {
            'BUY_PREPAID': _real_nodes.get_offers_node,
            'ASK_RECOMMENDATION': _real_nodes.recommend_offer_node,
            'RECOMMEND_DEVICE': _real_nodes.get_devices_node,
            'SUMMARIZE': _real_nodes.summary_node,
        }.get(state.get('intent'))
        if chosen:
            state.update(chosen(state))
        return state
    graph.recommendation_graph = SimpleNamespace(invoke=_invoke)
    sys.modules['recommendation_graph'] = graph

import communication_agent as ca
import nodes
from llm_response_generator import (
    ResponseValidationError,
    _validate_recommendation_response,
    generate_user_response,
)
from tools import get_compatible_devices_mock


def _reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _empty_context(currency='USD'):
    # PRE_002 est une information de prix MINIMUM issue de l'Agent de recommandation,
    # et NON une offre éligible pour le budget de 2 USD.
    return {
        'current_request': {'scope': 'recommendation', 'intent': 'BUY_PREPAID'},
        'recommendation_state': {
            'selection_basis': 'no_eligible_offer',
            'budget': 2,
            'budget_currency': currency,
            'offers': [],
            'devices': [],
            'lowest_available_offer': {
                'offer_id': 'PRE_002', 'name': 'Eco Prepaid',
                'price_monthly': 8, 'currency': 'USD', 'data_gb': 5,
            },
        },
        'selected_offer': None,
        'purchase_state': {'status': 'OFFER_UNAVAILABLE', 'offer': None},
    }


class NoEligibleOfferTests(unittest.TestCase):
    def test_previous_selection_disappears_below_catalogue_minimum(self):
        state = {
            'message': 'je veux un forfait de 2$',
            'intent': 'BUY_PREPAID',
            'budget': 8,
            'budget_currency': 'USD',
            'selected_offer_id': 'PRE_002',
            'offers': [{'offer_id': 'PRE_002'}],
        }
        state.update(nodes.extract_context_node(state))
        result = nodes.get_offers_node(state)
        self.assertEqual(state['budget'], 2)
        self.assertEqual(state['budget_currency'], 'USD')
        self.assertEqual(result['offers'], [])
        self.assertIsNone(result['selected_offer_id'])
        self.assertIsNone(result['selected_offer_name'])
        self.assertEqual(result['selection_basis'], 'no_eligible_offer')
        self.assertEqual(result['lowest_available_offer']['price_monthly'], 8)
        self.assertEqual(result['lowest_available_offer']['currency'], 'USD')

    def test_currency_not_invented_if_new_budget_has_no_unit(self):
        state = {
            'message': 'mon nouveau budget est 2', 'intent': 'BUY_PREPAID',
            'budget': 8, 'budget_currency': 'USD',
        }
        self.assertEqual(nodes.extract_context_node(state)['budget_currency'], None)
        context = _empty_context(currency=None)
        issues = _validate_recommendation_response(
            'Aucun forfait ne correspond à votre budget de 2 USD.', context
        )
        self.assertIn('unsupported_price_currency_pair', issues)

    def test_two_dollars_no_longer_rejected_when_no_offer(self):
        result = _validate_recommendation_response(
            'Aucun forfait ne correspond à votre budget de 2 USD. Le moins cher coûte 8 USD.',
            _empty_context(),
        )
        self.assertEqual(result, [])

    def test_wrong_unit_and_invented_minimum_still_rejected(self):
        context = _empty_context()
        self.assertIn('unsupported_currency:EUR', _validate_recommendation_response(
            'Aucun forfait pour votre budget de 2 EUR.', context))
        self.assertIn('unsupported_price_currency_pair', _validate_recommendation_response(
            'Nos offres commencent à 3 USD.', context))

    def test_generator_retries_wrong_euro_on_no_match(self):
        model = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=[
                _reply('Aucun forfait ne correspond à 2 EUR.'),
                _reply('Aucun forfait ne correspond à votre budget de 2 USD. Le moins cher coûte 8 USD.'),
            ])
        )))
        answer = generate_user_response(
            model, user_message='forfait 2$', business_context=_empty_context(),
            history=[], preferred_language='fr',
        )
        self.assertIn('2 USD', answer)
        self.assertNotIn('EUR', answer)
        self.assertEqual(model.chat.completions.create.call_count, 2)

    def test_generator_produces_single_line_with_no_match(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(return_value=_reply(
                'Aucun forfait ne correspond à votre budget de 2 USD.\nLe moins cher coûte **8 USD**.'
            )))))
        answer = generate_user_response(
            client,
            user_message='je veux un forfait de 2$',
            business_context=_empty_context(),
            history=[],
            preferred_language='fr',
        )
        self.assertEqual(answer, 'Aucun forfait ne correspond à votre budget de 2 USD. Le moins cher coûte 8 USD.')
        self.assertEqual(client.chat.completions.create.call_count, 1)
        for forbidden in ('\n', '\\n', '**', '||', '___'):
            self.assertNotIn(forbidden, answer)


class DeviceGuardTests(unittest.TestCase):
    def setUp(self):
        self.xiaomi = get_compatible_devices_mock.invoke({
            'offer_id': 'PRE_002', 'category_preference': 'SMARTPHONE',
        })[0]
        self.samsung = get_compatible_devices_mock.invoke({
            'offer_id': 'PRE_001', 'category_preference': 'SMARTPHONE',
        })[0]
        self.context = {
            'current_request': {'scope': 'recommendation', 'intent': 'RECOMMEND_DEVICE'},
            'recommendation_state': {
                'selected_offer_id': 'PRE_002', 'offers': [],
                'devices': [self.xiaomi], 'closest_device': None,
            },
            'selected_offer': None, 'purchase_state': {},
            '_validation_device_catalog': [self.xiaomi, self.samsung],
        }

    def test_samsung_plus_xiaomi_is_rejected_for_eco(self):
        answer = 'Le Samsung Galaxy A15 5G et le Xiaomi Redmi Note 13 sont recommandés.'
        self.assertIn('unlisted_device_mentioned', _validate_recommendation_response(answer, self.context))

    def test_xiaomi_alone_remains_valid(self):
        answer = 'Je recommande le Xiaomi Redmi Note 13 à 150 USD.'
        self.assertEqual(_validate_recommendation_response(answer, self.context), [])

    def test_generator_retries_invented_samsung_without_exposing_guard_catalogue(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=[
                _reply('Le Samsung Galaxy A15 5G et le Xiaomi Redmi Note 13 sont recommandés.'),
                _reply('Je recommande le Xiaomi Redmi Note 13 à 150 USD.'),
            ]))))
        visible = {key: value for key, value in self.context.items() if not key.startswith('_')}
        result = generate_user_response(
            client, user_message='un smartphone pour Ruben ?',
            business_context=visible,
            history=[], preferred_language='fr',
            validation_device_catalog=[self.xiaomi, self.samsung],
        )
        self.assertEqual(result, 'Je recommande le Xiaomi Redmi Note 13 à 150 USD.')
        self.assertEqual(client.chat.completions.create.call_count, 2)
        prompt = client.chat.completions.create.call_args_list[0].kwargs['messages'][1]['content']
        self.assertNotIn('Samsung Galaxy', prompt, 'Le catalogue de contrôle est invisible du verbaliseur')


class IntegratedConversationTests(unittest.TestCase):
    def setUp(self):
        ca._CONVERSATION_STATE.clear()
        ca._CONVERSATION_STATE.update({
            'customer_id': None, 'profile': {}, 'history': [],
            'last_intent': None, 'current_intent': None,
            'current_scope': 'unknown', 'recommendation': {},
            'tobi': {}, 'purchase': {},
        })

    def test_purchase_device_then_two_dollar_request_in_same_conversation(self):
        messages = [
            json.dumps({'intent': 'BUY_PREPAID', 'purchase_action': 'request',
                        'beneficiary_name': 'Ruben', 'recommendation_target': 'none'}),
            'Confirmez-vous cet achat simulé de Eco Prepaid à 8 USD pour Ruben?',
            json.dumps({'intent': 'BUY_PREPAID', 'purchase_action': 'confirm',
                        'beneficiary_name': None, 'recommendation_target': 'none'}),
            "L'achat simulé du forfait Eco Prepaid pour Ruben a été effectué avec succès.",
            json.dumps({'intent': 'RECOMMEND_DEVICE', 'purchase_action': 'none',
                        'beneficiary_name': None, 'recommendation_target': 'device'}),
            'Je recommande le Xiaomi Redmi Note 13 à 150 USD.',
            json.dumps({'intent': 'BUY_PREPAID', 'purchase_action': 'request',
                        'beneficiary_name': None, 'recommendation_target': 'none'}),
            'Aucun forfait ne correspond à votre budget de 2 USD. Le moins cher coûte 8 USD.',
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=[_reply(text) for text in messages])
        )))
        first = ca.handle_user_message(client, customer_id='CUST_001',
            message='achète un forfait de 8$ pour Ruben')
        self.assertEqual(first['data']['purchase']['status'], 'AWAITING_CONFIRMATION')
        self.assertEqual(first['data']['recommendation']['selected_offer_id'], 'PRE_002')
        second = ca.handle_user_message(client, message='oui je confirme')
        self.assertEqual(second['data']['purchase']['status'], 'COMPLETED')
        third = ca.handle_user_message(client, message='quel smartphone pour Ruben ?')
        self.assertEqual(third['intent'], 'RECOMMEND_DEVICE')
        self.assertEqual([d['device_id'] for d in third['data']['recommendation']['devices']], ['DEV_102'])
        self.assertNotIn('Samsung', third['response'])
        fourth = ca.handle_user_message(client, message='je veux un forfait de 2$')
        self.assertEqual(fourth['data']['recommendation']['offers'], [])
        self.assertIsNone(fourth['data']['recommendation']['selected_offer_id'])
        self.assertEqual(fourth['data']['recommendation']['budget_currency'], 'USD')
        self.assertEqual(fourth['data']['purchase']['status'], 'OFFER_UNAVAILABLE')
        self.assertIn('2 USD', fourth['response'])
        self.assertEqual(len(ca.get_conversation_state()['history']), 8)
        self.assertEqual(client.chat.completions.create.call_count, 8)

    def test_first_two_dollar_request_never_executes_an_old_offer(self):
        model = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=[
                _reply(json.dumps({
                    'intent': 'BUY_PREPAID', 'purchase_action': 'request',
                    'beneficiary_name': None, 'recommendation_target': 'none',
                })),
                _reply('Aucun forfait ne correspond à votre budget de 2 USD. Le moins cher coûte 8 USD.'),
            ])
        )))
        with patch.object(ca, 'purchase_prepaid_mock') as action:
            result = ca.handle_user_message(
                model, customer_id='CUST_001', message='je veux un forfait de 2$'
            )
        self.assertIsNone(result['data']['recommendation']['selected_offer_id'])
        self.assertEqual(result['data']['recommendation']['offers'], [])
        self.assertEqual(result['data']['purchase']['status'], 'OFFER_UNAVAILABLE')
        self.assertIn('2 USD', result['response'])
        action.invoke.assert_not_called()

    def test_new_device_question_supersedes_unconfirmed_mock_purchase(self):
        # Un appareil demandé avant la confirmation ne doit pas relancer l'achat.
        with patch.object(ca, 'classify_intent', side_effect=[
            {'intent': 'BUY_PREPAID', 'purchase_action': 'request', 'beneficiary_name': 'Ruben'},
            {'intent': 'RECOMMEND_DEVICE'},
        ]), patch.object(ca, 'generate_user_response', return_value='Je recommande le Xiaomi Redmi Note 13.'), \
             patch.object(ca, 'purchase_prepaid_mock') as action:
            ca.handle_user_message(object(), message='achète Eco Prepaid pour Ruben à 8$')
            result = ca.handle_user_message(object(), message='quel smartphone pour Ruben ?')
        self.assertEqual(result['data']['purchase']['status'], 'SUPERSEDED')
        action.invoke.assert_not_called()


class ChatHttpContractTests(unittest.TestCase):
    def test_chat_exposes_purchase_and_response_validation_failure_as_502(self):
        import importlib
        # Le vrai SDK s'il est présent ne recevra aucune clé réelle : /chat est mocké.
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-only-placeholder'}):
            with patch('groq.Groq', return_value=object()):
                import main
                importlib.reload(main)
        from fastapi.testclient import TestClient
        test_client = TestClient(main.app)
        payload = {
            'customer_id': 'CUST_001', 'profile_status': 'REGISTERED',
            'intent': 'BUY_PREPAID', 'scope': 'recommendation', 'response': 'Aucune offre pour 2 USD.',
            'data': {
                'recommendation': {'budget': 2, 'budget_currency': 'USD', 'offers': [],
                                   'selected_offer_id': None},
                'purchase': {'status': 'OFFER_UNAVAILABLE', 'offer': None},
                'tobi': {},
            },
        }
        with patch.object(main, 'handle_user_message', return_value=payload):
            response = test_client.post('/chat', json={'message': 'forfait de 2$'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['state']['purchase']['status'], 'OFFER_UNAVAILABLE')
        self.assertEqual(response.json()['offers'], [])
        with patch.object(main, 'handle_user_message', side_effect=ResponseValidationError('bad output')):
            response = test_client.post('/chat', json={'message': 'forfait de 2$'})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('ResponseValidationError', response.text)


if __name__ == '__main__':
    unittest.main()
