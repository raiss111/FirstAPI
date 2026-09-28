ÉTAPE 19 — ACHAT DE FORFAIT PRÉPAYÉ SIMULÉ + RÉPONSE SIMPLE
============================================================

Cette livraison est basée sur :
- étape 17 (sélection et devise USD sur les offres),
- étape 18 (nettoyage du texte de réponse),
- dernière architecture à conversation unique de l'étape 16.

FICHIERS À REMPLACER (à la racine du projet)
---------------------------------------------
1. intent_agent.py
   L'intent BUY_PREPAID reste identique. Le résultat de classification ajoute,
   pour cet intent seulement, purchase_action (browse/request/confirm/cancel/none)
   et beneficiary_name facultatif. L'Intent Agent n'exécute rien.

2. communication_agent.py
   Orchestrateur de l'unique ConversationState. Garde purchase dans le State,
   recueille une confirmation explicite, transmet l'offre choisie par le graphe
   au service d'achat Mock. L'accès à ce State unique est sérialisé (RLock).

3. tobi_services_mock.py
   Ajout de purchase_prepaid_mock : valide confirmed=True, relit l'offre dans
   le catalogue Mock et renvoie un succès FICTIF, jamais un achat réel.

4. llm_response_generator.py
   Verbaliseur uniquement. Sortie response en une seule ligne de texte simple :
   pas de vrai/littéral saut de ligne, Markdown, '||', underscores, procédure,
   canaux inventés ou promesse d'achat réel. Le texte non conforme est refusé.

5. main.py
   Conserve POST /chat sans session_id, sans nouveau système de session.
   Affiche state.purchase pour le debug (status, offer, action_result).

TESTS
-----
Ajouter : test_prepaid_mock_purchase.py
Remplacer les anciens tests (leur contrat précédent autorisait des retours à
la ligne et des phrases sur les procédures) :
- test_recommendation_consistency.py
- test_plain_responses.py
Conserver test_single_conversation.py de l'étape 16.

COMMANDES (dans ton venv, après sauvegarde des anciens fichiers)
-----------------------------------------------------------------
python -m unittest test_prepaid_mock_purchase test_single_conversation test_recommendation_consistency test_plain_responses -v
uvicorn main:app --reload

SCÉNARIO À TESTER /docs -> POST /chat (MÊME PROCESSUS FASTAPI)
--------------------------------------------------------------
1. {"message": "Achète-moi un forfait de 15$."}
   Attendu : state.purchase.status = AWAITING_CONFIRMATION;
   offre = Flexi Data Max (15 USD). Pas d'exécution à ce stade.
   Exemple de réponse : Flexi Data Max coûte 15 USD par mois. Confirmez-vous cet achat simulé ?

2. {"message": "Oui, je confirme cet achat simulé."}
   Attendu : state.purchase.status = COMPLETED ;
   state.purchase.action_result.status = SIMULATED_SUCCESS ;
   state.purchase.action_result.real_transaction = false.
   Exemple de réponse : Votre achat simulé de Flexi Data Max à 15 USD a été effectué.

3. Nouvel essai 'oui' ne relance PAS l'action précédente.

4. Pour tester une annulation, redémarrer FastAPI puis :
   {"message": "Achète Eco Prepaid pour Marie, à 8$."}
   {"message": "Non, j'annule."}
   Attendu : status=CANCELLED, action_result=null, aucune action exécutée.

5. Pour tester la consultation sans achat, après redémarrage :
   {"message": "Montre-moi les forfaits disponibles."}
   Attendu : recommendation présente, state.purchase reste {}.

CONTRAINTES ET LIMITES
----------------------
- Aucun paiement, aucun débit, aucune activation et aucune API opérateur réelle.
- Pour un achat "pour Marie", seul le nom déclaré est stocké; aucune identité
  ni numéro de téléphone n'est vérifié (simulation académique seulement).
- Aucun changement de requirements.txt, du frontend, de recommendation_graph.py,
  nodes.py, tools.py, state.py, crm_mock.py ou du State TOBi existant.
- Nécessite tools.py de l'étape 17 (devise USD déclarée dans le catalogue).
- Une seule conversation en RAM par processus FastAPI. Redémarrer le processus
  réinitialise le contexte; --reload peut réinitialiser à chaque modification.
- Dans cet environnement les tests ont utilisé des substituts pour les SDK
  absents. Valider aussi dans ton venv avec Groq réel et LangGraph réel.

RETOUR ARRIÈRE
--------------
Arrêter FastAPI; restaurer les cinq anciens fichiers et les anciens tests;
redémarrer FastAPI. Aucune migration de données persistantes.
