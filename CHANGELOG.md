# Changelog

Toutes les évolutions notables de l'application WFO sont documentées ici.

## 2026-02-08

- Ajout d'une traçabilité d'exécution dans l'UI et les exports:
- Génération d'un `run_id` par lancement d'optimisation.
- Capture des empreintes SHA256 de la configuration et des résultats.
- Capture des métadonnées Git (branche, commit, remote, état dirty).
- Export d'un fichier `audit_trace.json` dans les ZIP de résultats.
- Affichage d'un panneau `Traçabilité du run` dans l'interface Streamlit.

- Renforcement de la lisibilité du moteur adaptatif:
- Ajout de docstrings et commentaires structurants dans `adaptive_optimization.py`.

## 2026-02-09

- Extension du package d'export pour exploitation statistique et rejeu:
- Sauvegarde de tous les trials par fenêtre (`optimization_trials`) côté moteur WFO/adaptatif.
- Export `all_trials.csv` + `trials/window_<id>.csv`.
- Export `window_info.csv` (dates IS/OOS, compteurs d'évaluations).
- Export `replay_manifest.json` (config + traçabilité + contexte de rejeu).
- Option UI `Package complet (rejeu + stats)` qui force `df.csv` en mode `full`.
- Chargement automatique des fichiers `all_trials.csv` et `window_info.csv` depuis un ZIP.
- Nouveau choix utilisateur du snapshot marché dans l'export:
- `manifest_only` (sans snapshot prix), `parquet_zstd` (compact), `csv_downsampled`, `csv_full`.
- Ajout d'un descripteur de source de données dans `replay_manifest.json` (path, taille, période, lignes chargées).
- Ajout de profils adaptatifs sélectionnables dans l'UI:
- `Smoke Test`, `Rapide`, `Équilibré`, `Robuste`, `Réactif`, `Conservateur`, `Exploratoire`, `Custom`.
- Application explicite du preset via bouton (pas d'auto-réécriture des champs).
- Panneau explicatif par profil (objectif, avantages, limites, spécificités, impact durée).
- Estimation de charge et durée (bougies, cycles, trials, ETA) selon période/timeframe configurées.
- Améliorations ergonomiques UI:
- Migration `use_container_width` vers `width` pour anticiper la dépréciation Streamlit.
- Clarification du mode adaptatif: `Optimization Method` affichée mais désactivée/explicitement non applicable.
- Carte de charge contextuelle en sidebar (`cycles/trials/ETA`) en mode adaptatif.
- Table `All Trials` filtrable et paginée (fenêtres, score min, taille de page).
- Application des profils adaptatifs passée en action explicite (bouton), sans auto-réécriture silencieuse.
- Ajout d'animations d'attente pendant l'optimisation:
- Carte de statut animée (progression, elapsed, ETA, cycle/fenêtre).
- Badge `RUNNING`, barre de progression shimmer et bouton `Stop WFO` avec pulse visuel.
- Nouveau volet `Adaptive Insights` dans les résultats:
- Graphiques de convergence des scores par cycle (best, médiane, IQR).
- Graphique de gap IS/OOS (return et sharpe) pour visualiser la généralisation.
- Graphique de compression de grille (combinaisons actives vs baseline + trials testés).
- Graphique d'évolution des poids relatifs par paramètre (top 8).
- Vue top valeurs historiques par paramètre (`top_values_by_parameter`).
- Ajout d'un guide d'interprétation directement sous chaque graphique adaptatif.
- Ajout du module MVP `Expert IA`:
- Nouveau package `expert/` (gateway LLM OpenAI-compatible, prompt builder, schema, analyzer, storage, service).
- Nouvel onglet UI `🤖 Expert IA` pour générer une interprétation IA des résultats d'optimisation.
- Contrat JSON de sortie `expert.v1` avec validation minimale et mode dégradé en cas de réponse partielle.
- Sauvegarde des interprétations dans `reports/expert/` + journal `expert_audit.jsonl`.
- Mise en forme de l'analyse Expert pour lecture humaine dans l'UI:
- Remplacement de l'affichage JSON brut par un rapport structuré (évaluation globale, constats, généralisation, diagnostics adaptatifs, actions, alertes).
- Export utilisateur du rapport en Markdown (`.md`) au lieu d'un JSON brut.
- Édition des prompts Expert dans l'UI avant exécution:
- Affichage du prompt système et du prompt utilisateur dans un panneau éditable.
- Bouton `Charger prompts auto` pour régénérer le prompt par défaut à partir du contexte courant.
- Lancement de l'analyse avec les prompts réellement modifiés par l'utilisateur.
- Prompt codé mis à jour: réponse exigée en français avec tolérance aux anglicismes métier.
## 2026-02-10

- Upgrade Expert IA Phase 1:
- Prompt Expert orienté stratégie: injection du contexte entrée/sortie, modules de sortie actifs et rôle des paramètres.
- Ajout d'alertes déterministes (sans LLM) affichées avant l'analyse: overfitting IS/OOS, instabilité paramètres, insuffisance de trials, runs peu concluants.
- Ajout d'une gestion de templates de prompts dans l'UI Expert (sauvegarder/charger/supprimer).
- Rapport Expert enrichi avec section d'alertes déterministes et section d'alignement stratégie.
- Couverture explicite du final backtest dans l'Expert:
- Transmission des métriques finales (return, sharpe, drawdown, win rate, trades) et comparaison buy&hold quand disponibles.
- Affichage du résumé final backtest transmis dans l'UI Expert.
- Prompt mis à jour pour exiger une section d'analyse `final_backtest_assessment`.
- Sélection guidée des modèles LLM dans l'UI Expert:
- Ajout d'un catalogue de modèles par provider (`openai`, `grok`, `gemini`) avec descriptions métier.
- Sélecteur `Modèle LLM` basé sur liste + option `Autre (saisie libre)`.
- Placeholder/hint de `Base URL` adapté automatiquement au provider choisi.
- Correctifs de compatibilité OpenAI GPT-5 pour Expert IA:
- Gestion des paramètres API selon modèle (`max_completion_tokens` pour chat GPT-5, température personnalisée ignorée quand non supportée).
- Ajout du format JSON explicite dans les requêtes OpenAI (`response_format` / `text.format`) pour limiter les sorties non JSON.
- Détection explicite des réponses tronquées/incomplètes (`finish_reason=length`, `incomplete_details.reason=max_output_tokens`) avec message d'action.
- Prompt Expert rendu plus concis pour réduire le risque de troncature; `Max tokens` UI relevé à 3000 par défaut.
- Data Configuration UX:
- Ajout d'un sélecteur de fichier CSV (`Browse CSV file from disk`) pour choisir un fichier depuis le disque via l'interface.
- Le fichier uploadé est persisté en copie locale temporaire et renseigne automatiquement `File Path`.
- Synchronisation automatique des dates Start/End lors d'un nouvel upload.
- Correctifs robustesse métriques (audit Python):
- Correction du calcul `avg_pl_per_trade` pour les portefeuilles vectorisés (évite le test ambigu sur Series).
- Alignement du même correctif dans le moteur adaptatif.
- Documentation interne renforcée:
- Ajout de docstrings sur les fonctions principales (`app.py`, `data_loading.py`, `wfo.py`, `expert/*`, `config.py`) pour améliorer lisibilité et maintenance.
