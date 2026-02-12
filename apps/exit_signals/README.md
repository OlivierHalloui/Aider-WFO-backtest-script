# Exit Signals Apps

Ce dossier regroupe:
- `analyze_exit_signals.py`: analyse CLI d'un journal de trades TradingView.
- `exit_signal_app.py`: UI Streamlit pour piloter l'analyse.

## Lancement Streamlit

Depuis la racine du repo:

```bash
streamlit run apps/exit_signals/exit_signal_app.py
```

## Lancement CLI

```bash
python apps/exit_signals/analyze_exit_signals.py --help
```

## Dépendances minimales

```bash
pip install -r apps/exit_signals/requirements.txt
```
