# Analyse quant — prérequis du pair `wfo-quant`

Réf. : `docs/cahier_charges_analyse_quant.md` §6.2 (« Prérequis pair »).

L'onglet **Analyse quant** interroge un pair A2A local (`wfo-quant`) pour
produire l'interprétation qualitative d'un run WFO. Le pair n'est
**indispensable que pour le bloc §5.2 (interprétation)** : les calculs locaux
(manifeste, contrôle d'intégrité, indicateurs Q1–Q8, pré-verdict) fonctionnent
sans lui, en lecture seule.

## 1. Configuration attendue

Le profil `wfo-quant` doit exposer un point d'entrée A2A **local et sans token** :

| Réglage | Valeur exigée | Pourquoi |
|---|---|---|
| `platforms.a2a.enabled` | `true` | publie la carte d'agent A2A |
| `extra.port` | `9924` | port d'écoute (client : `http://127.0.0.1:9924`) |
| `approvals.mode` | **`off`** | obligatoire, voir ci-dessous |
| hôte | `127.0.0.1` (local uniquement) | condition du bypass d'authentification |

### Pourquoi `approvals.mode: off` est obligatoire

En session headless, le gate d'approbation du pair bloque ses outils
(« Silence is not consent ») et la requête revient en état **`input-required`**.
L'application détecte ce cas et affiche la marche à suivre au lieu de laisser
l'appel tourner dans le vide.

## 2. Vérification rapide

Le pair est sondé sur `/.well-known/agent-card.json` **avant** tout appel, et
**seulement** si le contrôle d'intégrité du run est OK (un run non évaluable ne
déclenche jamais d'aller réseau).

```bash
curl -s http://127.0.0.1:9924/.well-known/agent-card.json | head
```

Une réponse JSON confirme que le pair est joignable. Sans réponse :

1. vérifier que le profil `wfo-quant` est démarré ;
2. vérifier `platforms.a2a.enabled: true` et `extra.port: 9924` ;
3. vérifier `approvals.mode: off` ;
4. relancer depuis l'onglet **Analyse quant**.

## 3. Comportement sans pair

| Situation | Affichage | Appel A2A |
|---|---|---|
| Pair injoignable | bouton **inactif** + raison (sondage `agent-card`) | aucun |
| Contrôle d'intégrité en échec | `🔴 NO_GO` (pré-verdict local) + `non_evaluable — <cause>` | **aucun** (§5.3 cas a) |
| Run valide, pair joignable | bouton actif | appel `SendMessage`, timeout 900 s |

Le pré-verdict local (déterministe) reste affiché dans tous les cas : les
calculs §5.0/§5.1 ne dépendent jamais du pair.

**Convention utilisateur (dérogation explicite)** : les backtests sans
frais/slippage, y compris lorsque les deux champs sont absents, sont analysés
en **brut**. Cette absence est informative et ne bloque ni l'appel au pair ni
le pré-verdict C4 : un OOS brut positif peut satisfaire C4, un OOS nul ou
négatif reste `NO_GO`, un résultat manquant reste `WATCH`. Ne pas annoncer de
rentabilité nette ni faire de l'absence de coûts un motif bloquant récurrent.
Une valeur de coût déclarée mais malformée (y compris `None`) reste bloquante
pour l'intégrité, tout comme les autres défauts du run (§5.0).

## 4. Artefacts produits

Indépendamment du pair, un run donne :

- `run_manifest.json` — périmètre, provenance, résumé d'intégrité (§5.0) ;
- `quant_indicators.json` — indicateurs Q1–Q8, incertitude OOS (§5.1) ;
- pré-verdict local déterministe (§5.3).

Avec une analyse obtenue :

- `quant_analysis.json` — artefact **canonique** `quant_analysis.v1` (§5.2) ;
- `quant_analysis.md` — rendu dérivé, citations `{{ID.champ}}` résolues.

Les quatre fichiers sont joints au ZIP d'export, puis relus à l'import.

> **Archives, pas certificats.** Le ZIP porte les faits, pas les preuves brutes
> (`final_trades`, rendements par barre). À l'import, le contrôle d'intégrité est
> **recalculé** sur ce qui est disponible et signale ce qui manque — l'`ok: True`
> éventuellement stocké dans le fichier n'est jamais cru. Si le contrôle échoue,
> le pré-verdict local est `NO_GO` (§5.3 cas a) et aucun appel A2A n'est émis.
>
> Le champ `input_digest_replay` lie le manifeste au contenu du run et à la
> configuration du ZIP. Son absence ou un écart au réimport rejette les artefacts
> quant. Les diagnostics acceptés sont marqués `archived` et restent inchangés
> au rendu suivant ; le réexport conserve la configuration importée. Un nouveau
> contenu de run invalide l'archive et reprend le calcul local normal.
