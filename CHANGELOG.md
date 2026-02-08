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
