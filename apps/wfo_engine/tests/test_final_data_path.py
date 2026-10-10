"""Regression tests for the final-backtest source default."""
from collections import UserDict

from ui.data_utils import sync_final_data_file_path


def test_default_follows_wfo_source_changes():
    state = {}
    sync_final_data_file_path(state, '/data/A.csv')
    assert state['final_file_path'] == '/data/A.csv'
    sync_final_data_file_path(state, '/new-storage/A.csv')
    assert state['final_file_path'] == '/new-storage/A.csv'


def test_explicit_final_choice_is_preserved():
    state = {}
    sync_final_data_file_path(state, '/data/A.csv')
    state['final_file_path'] = '/holdout/B.csv'
    sync_final_data_file_path(state, '/data/C.csv')
    assert state['final_file_path'] == '/holdout/B.csv'


def test_clearing_final_choice_resumes_following():
    state = {'final_file_path': '/holdout/B.csv'}
    sync_final_data_file_path(state, '/data/A.csv')
    state['final_file_path'] = ''
    sync_final_data_file_path(state, '/data/C.csv')
    assert state['final_file_path'] == '/data/C.csv'


def test_existing_independent_final_choice_is_preserved():
    state = {'final_file_path': '/holdout/B.csv'}
    sync_final_data_file_path(state, '/data/A.csv')
    assert state['final_file_path'] == '/holdout/B.csv'


def test_legacy_upload_default_is_replaced_when_source_moves():
    state = {'final_file_path': '/old/uploads/A.csv',
             'uploaded_data_file_path': '/old/uploads/A.csv'}
    sync_final_data_file_path(state, '/new/data/A.csv')
    assert state['final_file_path'] == '/new/data/A.csv'


def test_empty_source_does_not_erase_final_choice():
    state = {'final_file_path': '/holdout/B.csv'}
    sync_final_data_file_path(state, '')
    assert state['final_file_path'] == '/holdout/B.csv'


def test_explicit_choice_equal_to_old_default_is_preserved():
    state = {}
    sync_final_data_file_path(state, '/data/A.csv')
    state['_final_file_path_explicit'] = True
    sync_final_data_file_path(state, '/data/B.csv')
    assert state['final_file_path'] == '/data/A.csv'


def test_streamlit_reruns_follow_source_and_preserve_manual_choice():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
from ui.data_utils import sync_final_data_file_path
source = st.text_input("File Path", value="/data/A.csv", key="file_path")
sync_final_data_file_path(st.session_state, source)
st.text_input("Final Data File Path", key="final_file_path",
              on_change=lambda: st.session_state.__setitem__(
                  "_final_file_path_explicit", bool(st.session_state.get("final_file_path"))))
''').run()
    assert not app.exception
    assert app.text_input[1].value == '/data/A.csv'
    app.text_input[0].set_value('/new/data/A.csv').run()
    assert not app.exception
    assert app.text_input[1].value == '/new/data/A.csv'
    app.text_input[1].set_value('/holdout/B.csv').run()
    app.text_input[0].set_value('/data/C.csv').run()
    assert not app.exception
    assert app.text_input[1].value == '/holdout/B.csv'
    app.text_input[1].set_value('').run()
    assert not app.exception
    assert app.text_input[1].value == '/data/C.csv'


def test_mapping_proxy_is_supported():
    state = UserDict()
    sync_final_data_file_path(state, '/data/A.csv')
    sync_final_data_file_path(state, '/data/B.csv')
    assert state['final_file_path'] == '/data/B.csv'
