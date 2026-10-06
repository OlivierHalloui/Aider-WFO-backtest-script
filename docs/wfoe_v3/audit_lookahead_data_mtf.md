# Audit lookahead — pipeline données + MTF (premier lot wfo-data)

Date : 2026-09-26. Auteur de l'audit : pair A2A wfo-data. Vérification et consolidation : orchestrateur wfo-engine.

Périmètre audité : pine_v3/mtf.py (144 l.) + pine_v3/mtf_parity.py (120 l.).
Statut de vérification : références fichier:ligne contrôlées contre le code réel par l'orchestrateur ; le point n°1 a été confirmé empiriquement (pandas 2.3.2).

## Synthèse

3 bloquants, 2 majeurs, 3 mineurs. Le trio bloquant tient à la fréquence Pine brute non convertie (pandas `'m'` = mois, pas minutes) et à la porte ouverte au `lookahead_on` en backtest.

## Points bloquants

1. `mtf.py:30` — resample avec fréquence Pine brute. CONFIRMÉ (test pandas 2.3.2 : `resample('15m')` → 1 barre au 31/01 ; `resample('15min')` → 2 barres correctes). La chaîne Pine (`'1m'`, `'5m'`, `'15m'`…, cf. app.py:1184) est passée telle quelle à `df.resample(timeframe)`. Pandas interprète `m` = month-end → la série HTF s'effondre en un bucket mensuel dont le Close est connu en fin de période = fuite massive + MTF incohérent. Correctif : convertir via `_to_pandas_freq` AVANT resample (la fonction existe déjà dans data_loading.py:72 et y est utilisée lignes 185/228, mais n'est PAS appliquée dans mtf.py).

2. `mtf.py:73-78` — `source_freq`/`target_freq` passées brutes à `vbt.Resampler` : même bug de fréquence, binning temporel faux, valeurs HTF attribuées à de mauvaises barres base. Correctif : mêmes fréquences normalisées en entrée du Resampler.

3. `mtf.py:82-85` — `timing="opening"` → `realign_opening` place la valeur HTF (ex. Close) à l'ouverture de sa barre : la clôture future devient disponible dès l'ouverture = repaint volontaire (fidèle à Pine `barmerge.lookahead_on`, mais contaminant en backtest sans alarme). Correctif : interdire ou tracer `lookahead_on` en backtest (forcer `closing`).

## Points majeurs

4. `mtf_parity.py:53-56` — les contrôles par appel (`non_na_count>0`, `change_count>0`, `base_bar_count>0`) ne vérifient que la présence de données, jamais la sémantique temporelle. Un script avec `lookahead_on` ou un mauvais alignement closing passe la « preuve de parité MTF » sans drapeau. Correctif : contrôle bloquant sur `lookahead_on` + assertion que les N premières barres base sont NaN en mode `closing`.

5. `mtf_parity.py:24-25` — `uses_mtf` dérivé de `capabilities.uses_request_security` déclarée ; si le spec omet cette capability alors qu'il contient un `request.security`, tous les checks MTF sont court-circuités (`passed=True` par défaut, lignes 44 et 75). Correctif : détecter la présence réelle de `request.security` dans les assignments, pas la capability déclarée.

## Points mineurs

6. `mtf.py:41-42` — `if isinstance(gaps, bool): return bool(gaps)` : un appelant passant `gaps=True` (croyant « gaps_on ») active en réalité `ffill=True` (comportement `gaps_off`). Inversion sémantique possible. Correctif : rejeter le booléen, n'accepter que `'on'`/`'off'`.

7. `mtf.py:143-144` — `request_security_signal` fait `fillna(False)` : un « pas encore connu » (NaN avant 1re clôture HTF) devient `False`, neutre en `AND` mais faux en contexte `OR`/sortie. Correctif : propager NaN, ne forcer `False` qu'en contexte `AND` explicite.

8. `mtf.py:69` — `series.sort_index()` sans dédup ni contrôle de monotonie : trous/doublons d'index HTF non corrigés avant `realign`. Correctif : `drop_duplicates` + assertion de monotonie avant alignement.

## À faire (suite du lot)

- Lot 2 : audit data_loading.py + ui/data_utils.py (même méthode).
- Lot 3 : section request.security/barmerge de runtime_adapter.py (~1567-1770).
- Ticket bloquant n°1 : voir board Kanban wfo-engine.
