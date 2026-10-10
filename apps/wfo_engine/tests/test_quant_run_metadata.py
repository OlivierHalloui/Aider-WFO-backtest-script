"""Regression probes derived from the real WFO export with nested metadata."""
import copy
import pytest

from services.quant_indicators import build_run_manifest
from ui.quant_analysis_panel import analysis_matches_run
from services.export_utils import quant_artifacts_for_run, restore_quant_artifacts


def real_run_shape():
    """Keep the production metadata shape without depending on a private ZIP."""
    return {
        'settings': {'n_windows': 5, 'optimization_method': 'bayesian'},
        'traceability': {'run': {'run_id': 'wfo-20261010T095833102373Z',
                                 'status': 'completed'}},
        'window_results': [{'optimization_trials': [], 'window_info': {'window': i}}
                           for i in range(1, 6)],
    }


def test_final_period_uses_final_dates_through_shared_diagnostics_path():
    from types import SimpleNamespace
    import pandas as pd
    from ui.quant_analysis_panel import ensure_quant_diagnostics

    portfolio = SimpleNamespace(trades=SimpleNamespace(
        records_readable=pd.DataFrame({'Return': [0.01], 'PnL': [1.0]})))
    state = {'final_start_date': '2026-01-01', 'final_end_date': '2026-01-14'}
    diagnostics, _, _ = ensure_quant_diagnostics(
        state, wfo_results=real_run_shape(),
        config={'start_date': '2026-01-01', 'end_date': '2026-01-14'},
        final_portfolio=portfolio,
    )
    assert diagnostics['manifest']['final_period'] == {
        'start': '2026-01-01', 'end': '2026-01-14',
    }


def test_q1_uses_explicit_config_when_run_budget_is_not_recorded():
    from services.quant_indicators import compute_q1
    q1 = compute_q1(real_run_shape(), config={'max_trials': 100})
    assert q1['demanded']['value'] == 500


def test_completed_final_period_survives_widget_edits():
    from services import export_utils
    portfolio = object()
    state = {}
    export_utils.record_final_quant_context(
        state, portfolio, final_start_date='2026-01-01', final_end_date='2026-01-14',
        final_file_path='/data/actual.csv', final_timeframe='5s',
    )
    state.update(final_start_date='2026-02-01', final_end_date='2026-02-14')
    effective = export_utils.quant_config_with_final_context({}, state, portfolio)
    assert effective['final_start_date'] == '2026-01-01'
    assert effective['final_end_date'] == '2026-01-14'
    assert effective['final_file_path'] == '/data/actual.csv'
    assert export_utils.quant_config_with_final_context({}, state, object())['final_start_date'] == '2026-02-01'


def test_diagnostics_cache_invalidates_when_indicator_version_changes(monkeypatch):
    from services import quant_indicators
    from ui.quant_analysis_panel import ensure_quant_diagnostics
    state = {}
    first, _, _ = ensure_quant_diagnostics(state, wfo_results=real_run_shape())
    monkeypatch.setattr(quant_indicators, 'INDICATOR_VERSION', 'metadata-fix-probe')
    second, _, _ = ensure_quant_diagnostics(state, wfo_results=real_run_shape())
    assert second is not first
    assert second['manifest']['indicator_version'] == 'metadata-fix-probe'


@pytest.mark.parametrize('budget', [None, 0, -1, 2.5, float('nan'), True, 'bad'])
def test_invalid_recorded_budget_never_falls_back_to_widget(budget):
    from services.quant_indicators import compute_q1
    run = real_run_shape()
    run['settings']['max_trials'] = budget
    demanded = compute_q1(run, config={'max_trials': 100})['demanded']
    assert demanded['value'] is None
    assert demanded['available'] is False
    assert demanded['raison']


def test_missing_budget_is_unavailable_without_default_200():
    from services.quant_indicators import compute_q1
    assert compute_q1(real_run_shape())['demanded']['value'] is None


def test_recorded_budget_wins_over_changed_widget():
    from services.quant_indicators import compute_q1
    run = real_run_shape()
    run['settings']['max_trials'] = 100
    assert compute_q1(run, config={'max_trials': 900})['demanded']['value'] == 500


def test_unknown_final_period_is_not_invented_from_wfo_dates():
    manifest = build_run_manifest(real_run_shape(), config={'end_date': '2026-01-14'})
    assert manifest['final_period'] == {'start': None, 'end': None}


def test_nested_run_id_is_checked_at_replay():
    run = real_run_shape()
    manifest = build_run_manifest(run)
    manifest['run_id'] = 'foreign-run'
    assert restore_quant_artifacts({'run_manifest.json': manifest}, run)['diagnostics'] is None


def test_replay_manifest_uses_executed_period_not_edited_widgets(monkeypatch):
    from types import SimpleNamespace
    from services.export_utils import record_final_quant_context, quant_config_with_final_context
    from ui import export_panel
    portfolio = object()
    state = {'final_start_date': '2026-02-01', 'final_end_date': '2026-02-14'}
    record_final_quant_context(state, portfolio, final_start_date='2026-01-01',
                               final_end_date='2026-01-14')
    config = quant_config_with_final_context({}, state, portfolio)
    monkeypatch.setattr(export_panel, 'st', SimpleNamespace(session_state=state))
    manifest = export_panel._build_replay_manifest({'config': config},
                                                   'manifest_only', 'manifest_only', False, {})
    assert manifest['final_backtest']['final_start_date'] == '2026-01-01'
    assert manifest['final_backtest']['final_end_date'] == '2026-01-14'


@pytest.mark.parametrize('with_archive', [False, True])
def test_import_new_run_clears_previous_final_context_and_cache(monkeypatch, with_archive):
    from types import SimpleNamespace
    from services.export_utils import apply_quant_restore, record_final_quant_context
    from services.quant_indicators import compute_quant_indicators
    from ui.quant_analysis_panel import ensure_quant_diagnostics
    from ui import export_panel

    old_portfolio = object()
    state = {'final_portfolio': old_portfolio, 'quant_diag_key': 'old-cache',
             'quant_diagnostics': {'manifest': {'run_id': 'OLD'}},
             'final_start_date': '2026-01-01', 'final_end_date': '2026-01-14'}
    record_final_quant_context(state, old_portfolio, final_start_date='2026-01-01',
                               final_end_date='2026-01-14')
    run = real_run_shape()
    run['traceability']['run']['run_id'] = 'NEW'
    config = {'max_trials': 100, 'final_start_date': '2026-02-01',
              'final_end_date': '2026-02-14'}
    diagnostics = compute_quant_indicators(run, config=config)
    restored = restore_quant_artifacts({
        'run_manifest.json': diagnostics['manifest'],
        'quant_indicators.json': diagnostics,
    } if with_archive else {}, run, config=config)
    apply_quant_restore(state, restored)
    assert state.get('final_portfolio') is None
    assert state.get('quant_final_context') is None
    assert state.get('quant_diag_key') is None
    current, _, _ = ensure_quant_diagnostics(
        state, wfo_results=run, config=config,
        final_portfolio=state.get('final_portfolio'),
    )
    assert current['manifest']['run_id'] == 'NEW'
    assert current['manifest']['final_period'] == {
        'start': '2026-02-01', 'end': '2026-02-14',
    }
    again, _, _ = ensure_quant_diagnostics(state, wfo_results=run, config=config)
    assert again is current
    import json
    import zipfile
    import pandas as pd
    state['wfo_results'] = run
    monkeypatch.setattr(export_panel, 'st', SimpleNamespace(
        session_state=state, warning=lambda message: None,
        error=lambda message: pytest.fail(message),
    ))
    empty = lambda *args, **kwargs: {}
    archive = export_panel._export_results_zip(
        build_results_payload=lambda: {'wfo_results': run, 'config': config},
        build_expert_context_pack_for_export=empty,
        compute_pine_order_semantics_report=empty,
        build_pine_beta_readiness_report=empty,
        build_pine_execution_gate_report=empty,
        resolve_pine_source_for_artifacts=empty,
        resolve_pine_libraries_for_artifacts=lambda: [],
        build_pine_generation_trace=empty,
        build_pine_artifacts_manifest=empty,
        build_window_info_dataframe=lambda results: pd.DataFrame(),
        build_trials_dataframe_from_results=lambda results: pd.DataFrame(),
    )
    with zipfile.ZipFile(archive) as zf:
        exported = json.loads(zf.read('run_manifest.json'))
        assert exported['run_id'] == 'NEW'
        assert exported['final_period'] == current['manifest']['final_period']
        assert 'final_trades.csv' not in zf.namelist()
        replay = json.loads(zf.read('replay_manifest.json'))
    assert replay['final_backtest']['final_start_date'] == '2026-02-01'
    assert replay['final_backtest']['final_end_date'] == '2026-02-14'


def test_traceability_run_id_reaches_manifest_and_export_guard():
    run = real_run_shape()
    before = copy.deepcopy(run)
    manifest = build_run_manifest(run, config={'max_trials': 100})
    assert manifest['run_id'] == 'wfo-20261010T095833102373Z'
    analysis = {'run_id': manifest['run_id'], 'input_digest': manifest['input_digest']}
    diagnostics = {'manifest': manifest}
    assert analysis_matches_run(analysis, diagnostics)
    assert quant_artifacts_for_run(analysis, diagnostics) is not None
    assert restore_quant_artifacts({'run_manifest.json': manifest}, run,
                                   config={'max_trials': 100})['diagnostics'] is not None
    assert run == before
