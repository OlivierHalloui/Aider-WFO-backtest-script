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
- Auto-application du preset au changement de profil + bouton de réapplication.
- Panneau explicatif par profil (objectif, avantages, limites, spécificités, impact durée).
- Estimation de charge et durée (bougies, cycles, trials, ETA) selon période/timeframe configurées.
