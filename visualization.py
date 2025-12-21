# Import necessary libraries for visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import io
from PIL import Image
import platform
import subprocess
import datetime
import os

# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_wfo_results(wfo_results, df):
    """
    Enhanced visualization of Walk-Forward Optimization results with explicit legends
    and descriptions for each chart.
    
    Parameters:
    -----------
    wfo_results : dict
        WFO results from walk_forward_optimization
    df : pandas.DataFrame
        Original OHLCV DataFrame
    """
    if not wfo_results['out_of_sample_performance']:
        print("No out-of-sample results to visualize")
        return
    
    # Set Seaborn style
    sns.set(style="whitegrid", font_scale=1.1)
    
    # Convert performance metrics to DataFrame
    oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    params_df = pd.DataFrame(wfo_results['best_params'])
    
    # =========================================================================
    # 1. MAIN OVERVIEW PLOT - PRICE WITH WINDOW BOUNDARIES
    # =========================================================================
    
    # Create a figure showing price with vertical lines indicating window boundaries
    plt.figure(figsize=(14, 10))
    
    # Plot the close price
    if 'Close' in df.columns:
        price_series = df['Close']
    elif 'close' in df.columns:
        price_series = df['close']
    else:
        price_series = df.iloc[:, 0]  # As a last resort
    
    # Plot price series
    plt.plot(df.index, price_series, label='Price', color='#1f77b4')
    
    # Add vertical lines for window boundaries
    window_colors = {
        'train': 'green',
        'test': 'red'
    }
    
    # Add vertical lines for window boundaries
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        window_info = window_result['window_info']
        
        # Add vertical line at window start
        plt.axvline(x=pd.to_datetime(window_info['start_date']), 
                   color='black', linestyle='--', alpha=0.5)
        
        # Mark in-sample and out-of-sample regions
        if window_info['in_sample_start'] and window_info['in_sample_end']:
            plt.axvspan(
                pd.to_datetime(window_info['in_sample_start']),
                pd.to_datetime(window_info['in_sample_end']),
                alpha=0.15, color=window_colors['train'], 
                label=f'In-Sample {window_idx+1}' if window_idx == 0 else ""
            )
        
        if window_info['out_sample_start'] and window_info['out_sample_end']:
            plt.axvspan(
                pd.to_datetime(window_info['out_sample_start']),
                pd.to_datetime(window_info['out_sample_end']),
                alpha=0.15, color=window_colors['test'], 
                label=f'Out-of-Sample {window_idx+1}' if window_idx == 0 else ""
            )
    
    # Customize the plot
    plt.title('Price Chart with Walk-Forward Optimization Windows', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add description as text in the plot
    plt.figtext(0.5, 0.01, 
               "This chart shows the price series with in-sample (green) and out-of-sample (red) periods highlighted.\n"
               "Each vertical line represents the boundary between consecutive WFO windows.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    # =========================================================================
    # 2. IN-SAMPLE VS OUT-OF-SAMPLE PERFORMANCE COMPARISON
    # =========================================================================
    
    # Extract performance metrics for both in-sample and out-of-sample
    is_metrics = []
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
            best_result = window_result['optimization_results'][0]
            if 'combined_score' in best_result:
                is_metrics.append({
                    'window': window_idx + 1,
                    'performance': best_result['combined_score'],
                    'type': 'In-Sample'
                })
    
    oos_metrics = []
    for metric in oos_df.to_dict('records'):
        oos_metrics.append({
            'window': metric['window'],
            'performance': metric['return'] if 'return' in metric else (
                          metric['sharpe'] if 'sharpe' in metric else 0),
            'type': 'Out-of-Sample'
        })
    
    comparison_df = pd.DataFrame(is_metrics + oos_metrics)
    
    if not comparison_df.empty:
        plt.figure(figsize=(14, 10))
        sns.barplot(
            x='window', 
            y='performance',
            hue='type',
            data=comparison_df,
            palette={'In-Sample': 'skyblue', 'Out-of-Sample': 'salmon'}
        )
        plt.title('In-Sample vs Out-of-Sample Performance by Window', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.figtext(0.5, 0.01, 
                   "This chart compares in-sample (optimization) performance with out-of-sample (validation) results.\n"
                   "Large differences suggest potential overfitting. Similar performance indicates strategy robustness.",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
    
    # =========================================================================
    # 3. PARAMETER STABILITY ANALYSIS
    # =========================================================================
    
    numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_params) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        norm_params_df = pd.DataFrame()
        for param in numeric_params:
            if params_df[param].nunique() <= 1:
                continue
            min_val = params_df[param].min()
            max_val = params_df[param].max()
            if max_val > min_val:
                norm_params_df[param] = (params_df[param] - min_val) / (max_val - min_val)
            else:
                norm_params_df[param] = params_df[param] / params_df[param]
        
        x = list(range(1, len(params_df) + 1))
        markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', '|', '_']
        for i, param in enumerate(norm_params_df.columns):
            marker = markers[i % len(markers)]
            plt.plot(x, norm_params_df[param], marker=marker, label=param, linewidth=2, markersize=8)
            stability = 1.0 - (np.std(params_df[param]) / np.mean(params_df[param])) if np.mean(params_df[param]) > 0 else 0.0
            plt.annotate(f"Stability: {stability:.2f}", 
                        xy=(x[-1], norm_params_df[param].iloc[-1]),
                        xytext=(x[-1] + 0.1, norm_params_df[param].iloc[-1]),
                        fontsize=9)
        
        plt.title('Parameter Stability Across Windows (Normalized Values)', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Normalized Parameter Value', fontsize=14)
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='best')
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.show()

def create_parameter_performance_map(wfo_results, top_n_params=5):
    if not wfo_results['window_results']: return
    all_top_params = []
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        if 'optimization_results' in window_result and window_result['optimization_results']:
            for i, param_set in enumerate(window_result['optimization_results'][:top_n_params]):
                if 'combined_score' not in param_set: continue
                param_dict = {k: v for k, v in param_set.items() 
                             if k not in ['combined_score', 'metric1_name', 'metric2_name', 
                                         'weight_metric1', 'weight_metric2']}
                param_key = ', '.join([f"{k}={v}" for k, v in sorted(param_dict.items())])
                all_top_params.append({
                    'window': window_idx + 1, 'rank': i + 1, 'param_key': param_key,
                    'score': param_set['combined_score'], **param_dict
                })
    if not all_top_params: return
    params_df = pd.DataFrame(all_top_params)
    param_counts = params_df['param_key'].value_counts().reset_index()
    param_counts.columns = ['Parameter Combination', 'Frequency']
    top_params = param_counts.head(min(10, len(param_counts)))
    plt.figure(figsize=(14, 10))
    plt.barh(top_params['Parameter Combination'], top_params['Frequency'], color='skyblue')
    plt.title('Most Frequent Top-Performing Parameter Combinations', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_robustness_metrics(wfo_results):
    if not wfo_results['out_of_sample_performance']: return
    oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    plt.figure(figsize=(10, 6))
    plt.plot(oos_df['window'], oos_df['return'], marker='o')
    plt.title('Out-of-Sample Returns by Window')
    plt.xlabel('Window')
    plt.ylabel('Return (%)')
    plt.grid(True)
    plt.show()

def generate_wfo_report_pdf(wfo_results, df, output_path=None, strategy_name="ATDMF Strategy"):
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"WFO_Report_{timestamp}.pdf"
    print(f"Saving report to {output_path}")
    # Minimal implementation for now to fix syntax
    with PdfPages(output_path) as pdf:
        plt.figure()
        plt.text(0.5, 0.5, f"Report for {strategy_name}")
        pdf.savefig()
        plt.close()
    return output_path

def integrate_report_generation(wfo_results, df, strategy_name="ATDMF Strategy"):
    """
    Wrapper function to integrate report generation.
    """
    try:
        report_path = generate_wfo_report_pdf(wfo_results, df, strategy_name=strategy_name)
        print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
