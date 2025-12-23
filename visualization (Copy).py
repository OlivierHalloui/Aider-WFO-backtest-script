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
    
    # Create a comparison of key metrics between in-sample and out-of-sample periods
    # This helps assess overfitting
    
    # Extract performance metrics for both in-sample and out-of-sample
    is_metrics = []
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        # Get best in-sample result from optimization results
        if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
            best_result = window_result['optimization_results'][0]
            if 'combined_score' in best_result:
                is_metrics.append({
                    'window': window_idx + 1,
                    'performance': best_result['combined_score'],
                    'type': 'In-Sample'
                })
    
    # Get out-of-sample metrics
    oos_metrics = []
    for metric in oos_df.to_dict('records'):
        oos_metrics.append({
            'window': metric['window'],
            'performance': metric['return'] if 'return' in metric else (
                          metric['sharpe'] if 'sharpe' in metric else 0),
            'type': 'Out-of-Sample'
        })
    
    # Combine metrics
    comparison_df = pd.DataFrame(is_metrics + oos_metrics)
    
    if not comparison_df.empty:
        plt.figure(figsize=(14, 10))
        
        # Create grouped bar chart
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
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart compares in-sample (optimization) performance with out-of-sample (validation) results.\n"
                   "Large differences suggest potential overfitting. Similar performance indicates strategy robustness.",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
    
    # =========================================================================
    # 3. PARAMETER STABILITY ANALYSIS
    # =========================================================================
    
    # This visualization shows how parameters change across windows
    # Stable parameters indicate robust strategies
    
    # Filter only numeric parameters
    numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_params) > 0:
        # Create a parameter stability plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get normalized parameter values for comparison across different scales
        norm_params_df = pd.DataFrame()
        for param in numeric_params:
            # Skip if all values are identical (stability = 1.0)
            if params_df[param].nunique() <= 1:
                continue
                
            # Normalize to 0-1 range for plotting on same scale
            min_val = params_df[param].min()
            max_val = params_df[param].max()
            
            if max_val > min_val:  # Avoid division by zero
                norm_params_df[param] = (params_df[param] - min_val) / (max_val - min_val)
            else:
                norm_params_df[param] = params_df[param] / params_df[param]
        
        # Create x-axis (window numbers)
        x = list(range(1, len(params_df) + 1))
        
        # Plot each parameter
        markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', '|', '_']
        for i, param in enumerate(norm_params_df.columns):
            marker = markers[i % len(markers)]
            plt.plot(x, norm_params_df[param], marker=marker, label=param, linewidth=2, markersize=8)
            
            # Calculate stability metric (1 - coefficient of variation)
            stability = 1.0 - (np.std(params_df[param]) / np.mean(params_df[param])) if np.mean(params_df[param]) > 0 else 0.0
            
            # Annotate with stability value
            plt.annotate(f"Stability: {stability:.2f}", 
                        xy=(x[-1], norm_params_df[param].iloc[-1]),
                        xytext=(x[-1] + 0.1, norm_params_df[param].iloc[-1]),
                        fontsize=9)
        
        # Add parameter absolute value table
        param_table = ''
        for i, window in enumerate(x):
            param_table += f'Window {window}: '
            param_values = []
            for param in numeric_params:
                if param in norm_params_df.columns:
                    param_values.append(f"{param}={params_df[param].iloc[i]:.2f}")
            param_table += ', '.join(param_values) + '\n'
        
        plt.figtext(0.5, 0.01, param_table, ha="center", fontsize=9, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # Customize the plot
        plt.title('Parameter Stability Across Windows (Normalized Values)', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Normalized Parameter Value', fontsize=14)
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)  # Give some margin above and below
        plt.legend(loc='best')
        
        # Add description
        plt.figtext(0.5, 0.15, 
                   "This chart shows how optimized parameters change across windows (normalized to 0-1 scale).\n"
                   "Stable parameters (less variation) indicate more robust strategies.\n"
                   "Stability score ranges from 0 (unstable) to 1 (perfectly stable).",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.25, 1, 0.97])
        plt.show()
    
    # =========================================================================
    # 4. OUT-OF-SAMPLE PERFORMANCE METRICS DASHBOARD
    # =========================================================================
    
    # Create a comprehensive dashboard of OOS performance metrics
    if not oos_df.empty:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns by Window (%)', 'Sharpe Ratio by Window', 
                          'Maximum Drawdown by Window (%)', 'Win Rate by Window (%)'),
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Returns plot
        if 'return' in oos_df.columns:
            window_nums = list(range(1, len(oos_df) + 1))
            
            # Bar chart with returns
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['return'],
                    name='Return (%)',
                    marker_color='rgb(55, 83, 109)',
                    text=oos_df['return'].round(2).astype(str) + '%',
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Add benchmark line (average return)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['return'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Return: {oos_df["return"].mean():.2f}%',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Sharpe ratio plot
        if 'sharpe' in oos_df.columns:
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['sharpe'],
                    name='Sharpe Ratio',
                    marker_color='rgb(26, 118, 255)',
                    text=oos_df['sharpe'].round(2).astype(str),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Add benchmark line (average Sharpe)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['sharpe'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Sharpe: {oos_df["sharpe"].mean():.2f}',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
        
        # 3. Max drawdown plot
        if 'max_drawdown' in oos_df.columns:
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['max_drawdown'],
                    name='Max Drawdown (%)',
                    marker_color='rgb(204, 0, 0)',
                    text=oos_df['max_drawdown'].round(2).astype(str) + '%',
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # Add benchmark line (average max drawdown)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['max_drawdown'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Drawdown: {oos_df["max_drawdown"].mean():.2f}%',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=1
            )
        
        # 4. Win rate plot
        if 'win_rate' in oos_df.columns:
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['win_rate'],
                    name='Win Rate (%)',
                    marker_color='rgb(60, 179, 113)',
                    text=oos_df['win_rate'].round(2).astype(str) + '%',
                    textposition='auto'
                ),
                row=2, col=2
            )
            
            # Add benchmark line (average win rate)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['win_rate'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Win Rate: {oos_df["win_rate"].mean():.2f}%',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text='Out-of-Sample Performance Metrics Dashboard',
            title_font_size=20,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800,
            width=1200,
            annotations=[
                dict(
                    text="This dashboard shows key performance metrics across all out-of-sample periods.<br>Consistent results across windows indicate strategy robustness.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    font=dict(size=14)
                )
            ]
        )
        
        # Update axes
        fig.update_xaxes(title_text='Window', row=2, col=1)
        fig.update_xaxes(title_text='Window', row=2, col=2)
        fig.update_yaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=2)
        fig.update_yaxes(title_text='Max Drawdown (%)', row=2, col=1)
        fig.update_yaxes(title_text='Win Rate (%)', row=2, col=2)
        
        fig.show()
    
    # =========================================================================
    # 5. PARAMETER IMPACT HEATMAP
    # =========================================================================
    
    # This visualization shows how different parameters impact performance metrics
    # Helps identify which parameters are most important for the strategy
    
    if len(numeric_params) >= 2 and len(oos_df) >= 3:
        # Compute correlation between parameters and metrics
        correlation_data = []
        
        for param in numeric_params:
            for metric in ['return', 'sharpe', 'max_drawdown', 'win_rate']:
                if metric in oos_df.columns and not oos_df[metric].isna().all():
                    # Skip if all parameter values are identical
                    if params_df[param].nunique() <= 1:
                        continue
                        
                    # Get valid values for correlation calculation
                    param_values = params_df[param].values
                    metric_values = oos_df[metric].values
                    valid_mask = ~np.isnan(param_values) & ~np.isnan(metric_values)
                    
                    if sum(valid_mask) >= 3:  # Need at least 3 valid points for meaningful correlation
                        try:
                            corr = np.corrcoef(param_values[valid_mask], metric_values[valid_mask])[0, 1]
                            if not np.isnan(corr) and not np.isinf(corr):
                                correlation_data.append({
                                    'Parameter': param,
                                    'Metric': metric,
                                    'Correlation': corr
                                })
                        except Exception as e:
                            print(f"Could not calculate correlation for {param} vs {metric}: {e}")
        
        # Create a correlation heatmap if we have data
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            # Pivot to create a matrix suitable for heatmap
            pivot_df = corr_df.pivot(index='Parameter', columns='Metric', values='Correlation')
            
            plt.figure(figsize=(14, 10))
            heatmap = sns.heatmap(
                pivot_df,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}
            )
            
            plt.title('Parameter Impact on Performance Metrics', fontsize=16)
            plt.ylabel('Parameter', fontsize=14)
            plt.xlabel('Performance Metric', fontsize=14)
            
            # Add description
            plt.figtext(0.5, 0.01, 
                       "This heatmap shows correlations between optimized parameters and out-of-sample performance metrics.\n"
                       "Positive correlations (blue) indicate that increasing the parameter tends to improve the metric.\n"
                       "Negative correlations (red) indicate that decreasing the parameter tends to improve the metric.",
                       ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.07, 1, 0.97])
            plt.show()
    
    # =========================================================================
    # 6. TRADES ANALYSIS VISUALIZATION
    # =========================================================================
    
    # This visualization compares trade metrics across windows
    if 'n_trades' in oos_df.columns and not oos_df['n_trades'].isna().all():
        fig, ax1 = plt.subplots(figsize=(14, 10))
        
        # X-axis: window numbers
        window_labels = [f"{i+1}" for i in range(len(oos_df))]
        x = np.arange(len(window_labels))
        
        # Plot number of trades as bars
        bars = ax1.bar(x - 0.2, oos_df['n_trades'], width=0.4, color='skyblue', label='Number of Trades')
        ax1.set_xlabel('Window', fontsize=14)
        ax1.set_ylabel('Number of Trades', fontsize=14, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Add trade count labels above bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
        
        # Plot win rate on the same graph with secondary y-axis
        ax2 = ax1.twinx()
        if 'win_rate' in oos_df.columns and not oos_df['win_rate'].isna().all():
            bars2 = ax2.bar(x + 0.2, oos_df['win_rate'], width=0.4, color='salmon', label='Win Rate (%)')
            ax2.set_ylabel('Win Rate (%)', fontsize=14, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add win rate labels above bars
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
        
        # Add a horizontal line for average win rate
        if 'win_rate' in oos_df.columns:
            avg_win_rate = oos_df['win_rate'].mean()
            ax2.axhline(y=avg_win_rate, color='red', linestyle='dashed', alpha=0.8, 
                       label=f'Avg Win Rate: {avg_win_rate:.2f}%')
        
        # Set x-ticks at bar positions
        ax1.set_xticks(x)
        ax1.set_xticklabels(window_labels)
        
        # Title and grid
        plt.title('Trade Metrics by Window', fontsize=16)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart compares the number of trades (blue) and win rate (red) across each window.\n"
                   "Consistent trade frequency and win rates indicate stable strategy performance.",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        plt.show()
    
    # =========================================================================
    # 7. DISTRIBUTION OF RETURNS (RISK ANALYSIS)
    # =========================================================================
    
    # This visualization shows the distribution of returns and key risk metrics
    if 'return' in oos_df.columns and not oos_df['return'].isna().all():
        plt.figure(figsize=(14, 10))
        
        # Create distribution plot with kernel density estimation
        sns.histplot(oos_df['return'].values, kde=True, stat="density", 
                   color='skyblue', bins=min(10, len(oos_df)))
        
        # Mark important statistics
        mean_return = oos_df['return'].mean()
        median_return = oos_df['return'].median()
        std_return = oos_df['return'].std()
        
        # Add vertical lines for mean, median
        plt.axvline(mean_return, color='red', linestyle='dashed', linewidth=2, 
                  label=f'Mean: {mean_return:.2f}%')
        plt.axvline(median_return, color='green', linestyle='dashed', linewidth=2, 
                   label=f'Median: {median_return:.2f}%')
        
        # Add vertical lines for mean ± 1 std dev (68% confidence interval)
        plt.axvline(mean_return + std_return, color='purple', linestyle='dotted', 
                  label=f'Mean + 1σ: {mean_return + std_return:.2f}%')
        plt.axvline(mean_return - std_return, color='purple', linestyle='dotted', 
                  label=f'Mean - 1σ: {mean_return - std_return:.2f}%')
        
        # Add horizontal line at y=0 to emphasize negative returns
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Calculate key metrics for the text box
        negative_returns = (oos_df['return'] < 0).mean() * 100
        positive_returns = (oos_df['return'] > 0).mean() * 100
        skewness = oos_df['return'].skew()
        kurtosis = oos_df['return'].kurtosis()
        
        # Add text box with metrics
        metrics_text = (
            f"Distribution Metrics:\n"
            f"Mean Return: {mean_return:.2f}%\n"
            f"Median Return: {median_return:.2f}%\n"
            f"Standard Deviation: {std_return:.2f}%\n"
            f"Negative Windows: {negative_returns:.1f}%\n"
            f"Positive Windows: {positive_returns:.1f}%\n"
            f"Skewness: {skewness:.2f}\n"
            f"Kurtosis: {kurtosis:.2f}"
        )
        
        # Add the metrics textbox
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    backgroundcolor='white', alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                    verticalalignment='top')
        
        # Customize the plot
        plt.title('Distribution of Out-of-Sample Returns', fontsize=16)
        plt.xlabel('Return (%)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart shows the distribution of out-of-sample returns across all windows.\n"
                   "A good strategy should have a distribution skewed to the right (positive returns)\n"
                   "with a majority of returns above zero.",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        plt.show()

def create_parameter_performance_map(wfo_results, top_n_params=5):
    """
    Creates a visual map showing which parameter combinations performed best
    across different windows, helping identify robust parameter sets.
    
    Parameters:
    -----------
    wfo_results : dict
        WFO results from walk_forward_optimization
    top_n_params : int, optional
        Number of top parameter combinations to analyze per window
        
    Returns:
    --------
    None
    """
    if not wfo_results['window_results']:
        print("No window results to analyze")
        return
    
    # Extract top parameter combinations for each window
    all_top_params = []
    
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        if 'optimization_results' in window_result and window_result['optimization_results']:
            # Get top N parameter sets
            for i, param_set in enumerate(window_result['optimization_results'][:top_n_params]):
                # Skip if no combined_score
                if 'combined_score' not in param_set:
                    continue
                
                # Extract parameters (excluding scores and metrics)
                param_dict = {k: v for k, v in param_set.items() 
                             if k not in ['combined_score', 'metric1_name', 'metric2_name', 
                                         'weight_metric1', 'weight_metric2']}
                
                # Create a parameter key (string representation of parameters)
                param_key = ', '.join([f"{k}={v}" for k, v in sorted(param_dict.items())])
                
                all_top_params.append({
                    'window': window_idx + 1,
                    'rank': i + 1,
                    'param_key': param_key,
                    'score': param_set['combined_score'],
                    **param_dict  # Include individual parameters
                })
    
    # Convert to DataFrame
    if not all_top_params:
        print("No parameter data available for visualization")
        return
        
    params_df = pd.DataFrame(all_top_params)
    
    # =========================================================================
    # 1. PARAMETER FREQUENCY ANALYSIS
    # =========================================================================
    
    # Count frequency of each parameter combination
    param_counts = params_df['param_key'].value_counts().reset_index()
    param_counts.columns = ['Parameter Combination', 'Frequency']
    
    # Get the top N most frequent parameter combinations
    top_params = param_counts.head(min(10, len(param_counts)))
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(top_params['Parameter Combination'], top_params['Frequency'], color='skyblue')
    
    # Add count labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f"{width}", va='center')
    
    plt.title('Most Frequent Top-Performing Parameter Combinations Across Windows', fontsize=16)
    plt.xlabel('Frequency (Number of Windows)', fontsize=14)
    plt.ylabel('Parameter Combination', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add description
    plt.figtext(0.5, 0.01, 
               "This chart shows which parameter combinations appeared most frequently in the top-performing results across windows.\n"
               "Parameter combinations that consistently perform well indicate more robust strategy settings.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    plt.show()
    
    # =========================================================================
    # 2. PARAMETER PERFORMANCE HEATMAP
    # =========================================================================
    
    # If we have any numeric parameters with multiple values, create 2D heatmaps
    numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
    # Filter out window, rank, and score columns
    numeric_params = [p for p in numeric_params if p not in ['window', 'rank', 'score']]
    
    if len(numeric_params) >= 2:
        # Get the two parameters with the most unique values
        param_unique_counts = [(param, params_df[param].nunique()) for param in numeric_params]
        param_unique_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top two parameters for the heatmap
        if len(param_unique_counts) >= 2:
            param_x, _ = param_unique_counts[0]
            param_y, _ = param_unique_counts[1]
            
            # Create a pivot table of average scores for parameter combinations
            pivot_data = params_df.pivot_table(
                values='score', 
                index=param_y,
                columns=param_x,
                aggfunc='mean'
            )
            
            plt.figure(figsize=(14, 10))
            heatmap = sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".2f",
                cmap='viridis',
                linewidths=0.5,
                cbar_kws={"shrink": .8, "label": "Average Performance Score"}
            )
            
            plt.title(f'Parameter Performance Map: {param_y} vs {param_x}', fontsize=16)
            plt.xlabel(param_x, fontsize=14)
            plt.ylabel(param_y, fontsize=14)
            
            # Add description
            plt.figtext(0.5, 0.01, 
                       f"This heatmap visualizes how different combinations of {param_x} and {param_y} impact performance.\n"
                       "Darker colors indicate better performance. This helps identify optimal parameter regions across all windows.",
                       ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.07, 1, 0.97])
            plt.show()
    
    # =========================================================================
    # 3. PARAMETER RANKING VISUALIZATION
    # =========================================================================
    
    # Create visualization of parameter ranks across windows
    # This shows if certain parameters consistently rank high
    
    # Get top 3 parameter combinations for each window
    top3_params = params_df[params_df['rank'] <= 3].copy()
    
    if len(top3_params) > 0:
        plt.figure(figsize=(14, 10))
        
        # Create a rank plot by window
        sns.scatterplot(
            data=top3_params,
            x='window',
            y='rank',
            hue='param_key',
            size='score',
            sizes=(100, 400),
            alpha=0.7,
            palette='viridis'
        )
        
        # Customize the plot
        plt.title('Top Parameter Rankings by Window', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Rank (1 = Best)', fontsize=14)
        plt.yticks([1, 2, 3])
        plt.grid(True, alpha=0.3)
        
        # Reverse y-axis so that rank 1 is at the top
        plt.gca().invert_yaxis()
        
        # If there are too many parameter combinations, limit the legend
        if len(top3_params['param_key'].unique()) > 10:
            # Get the most frequent parameter combinations for the legend
            top_params = top3_params['param_key'].value_counts().head(10).index.tolist()
            handles, labels = plt.gca().get_legend_handles_labels()
            
            # Create a filtered legend
            new_handles = []
            new_labels = []
            for handle, label in zip(handles, labels):
                if label in top_params or 'score' in label:
                    new_handles.append(handle)
                    new_labels.append(label)
            
            plt.legend(new_handles, new_labels, title='Parameter Combinations', 
                     loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        else:
            plt.legend(title='Parameter Combinations', loc='upper center', 
                     bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart shows which parameter combinations ranked in the top 3 for each window.\n"
                   "Parameter combinations that appear multiple times demonstrate consistent performance.\n"
                   "Marker size indicates performance score (larger = better).",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.2, 1, 0.97])
        plt.show()

def visualize_robustness_metrics(wfo_results):
    """
    Creates visualizations specifically focused on strategy robustness metrics
    derived from Walk-Forward Optimization results.
    
    Parameters:
    -----------
    wfo_results : dict
        WFO results from walk_forward_optimization
        
    Returns:
    --------
    None
    """
    if not wfo_results['out_of_sample_performance']:
        print("No out-of-sample results to analyze robustness")
        return
    
    # Extract out-of-sample performance data
    oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    
    # Extract parameter data
    params_df = pd.DataFrame(wfo_results['best_params'])
    
    # Define robustness metrics
    robustness_metrics = {}
    
    # 1. Performance Consistency: % of positive OOS periods
    if 'return' in oos_df.columns:
        positive_periods = (oos_df['return'] > 0).mean() * 100
        robustness_metrics['Positive Periods (%)'] = positive_periods
    
    # 2. Performance Stability: coefficient of variation of returns
    #    Lower values indicate more stable returns
    if 'return' in oos_df.columns and not oos_df['return'].isna().all():
        # Calculate coefficient of variation (std/mean) for non-zero mean
        mean_return = oos_df['return'].mean()
        if abs(mean_return) > 1e-6:  # Avoid division by zero or tiny numbers
            cv_return = oos_df['return'].std() / abs(mean_return)
            # Convert to stability (1 - normalized CV)
            # Limit to range [0, 1] by using 1 / (1 + CV)
            return_stability = 1 / (1 + cv_return)
            robustness_metrics['Return Stability (0-1)'] = return_stability
    
    # 3. Sharpe Consistency: coefficient of variation of Sharpe ratios
    if 'sharpe' in oos_df.columns and not oos_df['sharpe'].isna().all():
        # Only for positive mean Sharpe
        mean_sharpe = oos_df['sharpe'].mean()
        if mean_sharpe > 1e-6:
            cv_sharpe = oos_df['sharpe'].std() / mean_sharpe
            sharpe_stability = 1 / (1 + cv_sharpe)
            robustness_metrics['Sharpe Stability (0-1)'] = sharpe_stability
    
    # 4. Win Rate Consistency
    if 'win_rate' in oos_df.columns and not oos_df['win_rate'].isna().all():
        mean_win_rate = oos_df['win_rate'].mean()
        if mean_win_rate > 1e-6:
            cv_win_rate = oos_df['win_rate'].std() / mean_win_rate
            win_rate_stability = 1 / (1 + cv_win_rate)
            robustness_metrics['Win Rate Stability (0-1)'] = win_rate_stability
    
    # 5. Parameter Stability: average of 1 - CV for each parameter
    param_stability = {}
    numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
    
    for param in numeric_params:
        mean_value = params_df[param].mean()
        if abs(mean_value) > 1e-6:  # Avoid division by zero
            cv = params_df[param].std() / abs(mean_value)
            stability = 1 / (1 + cv)
            param_stability[param] = stability
    
    if param_stability:
        robustness_metrics['Avg Parameter Stability (0-1)'] = np.mean(list(param_stability.values()))
        # Also store individual parameter stability
        for param, stability in param_stability.items():
            robustness_metrics[f'{param} Stability'] = stability
    
    # 6. OOS vs IS Performance Ratio 
    # If we have both IS and OOS metrics, calculate ratio
    is_metrics = []
    for window_result in wfo_results['window_results']:
        if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
            best_result = window_result['optimization_results'][0]
            if 'combined_score' in best_result:
                is_metrics.append(best_result['combined_score'])
    
    if is_metrics and 'return' in oos_df.columns:
        is_avg = np.mean(is_metrics)
        oos_avg = oos_df['return'].mean()
        
        if abs(is_avg) > 1e-6:  # Avoid division by zero
            oos_is_ratio = oos_avg / is_avg
            # Normalize to 0-1 scale: 1 means OOS = IS (perfect), 0 means completely different
            oos_is_consistency = 1 - min(1, abs(1 - oos_is_ratio))
            robustness_metrics['OOS/IS Consistency (0-1)'] = oos_is_consistency
    
    # 7. Calculate overall robustness score (average of all metrics)
    # Filter to only include 0-1 scaled metrics
    scaled_metrics = {k: v for k, v in robustness_metrics.items() if '(0-1)' in k}
    if scaled_metrics:
        robustness_metrics['Overall Robustness Score (0-1)'] = np.mean(list(scaled_metrics.values()))
    
    # =========================================================================
    # VISUALIZATION: ROBUSTNESS DASHBOARD
    # =========================================================================
    
    plt.figure(figsize=(14, 10))
    
    # Filter metrics to only include those on 0-1 scale for the radar chart
    radar_metrics = {k.replace(' (0-1)', ''): v for k, v in robustness_metrics.items() if '(0-1)' in k}
    
    if len(radar_metrics) >= 3:
        # Create radar chart (spider plot) for robustness metrics
        # Prepare data for radar chart
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        
        # Close the plot by appending the first value at the end
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # Calculate angle for each category
        N = len(categories) - 1  # Excluding the repeated first element
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create radar plot
        ax = plt.subplot(2, 2, 1, polar=True)
        
        # Draw the chart
        plt.polar(angles, values, marker='o', linestyle='-', linewidth=2, label='Robustness Metrics')
        
        # Fill the area
        plt.fill(angles, values, alpha=0.25)
        
        # Set category labels
        plt.xticks(angles[:-1], categories[:-1])
        
        # Set radial limits
        plt.ylim(0, 1)
        
        # Add radial grid lines at 0.2, 0.4, 0.6, 0.8
        plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
        
        # Draw y-axis circles
        for ytick in [0.2, 0.4, 0.6, 0.8]:
            ax.add_artist(plt.Circle((0, 0), ytick, fill=False, color='grey', linestyle='--', alpha=0.4))
        
        # Title for this subplot
        plt.title('Strategy Robustness Metrics (1.0 = Ideal)', fontsize=14, pad=20)
    
    # Create bar chart with all robustness metrics
    ax2 = plt.subplot(2, 2, 2)
    
    metrics_to_plot = {k: v for k, v in robustness_metrics.items() 
                     if not k.startswith('Avg') and '(0-1)' in k}
    
    # Sort metrics by value
    sorted_metrics = dict(sorted(metrics_to_plot.items(), key=lambda item: item[1], reverse=True))
    
    # Plot bar chart
    bars = plt.barh(
        [k.replace(' (0-1)', '') for k in sorted_metrics.keys()], 
        list(sorted_metrics.values()),
        color='skyblue'
    )
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
               f'{width:.2f}', va='center')
    
    plt.xlim(0, 1.1)
    plt.xlabel('Score (0-1 Scale)', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.title('Robustness Metrics Comparison', fontsize=14)
    plt.grid(True, axis='x', alpha=0.3)
    
    # Create parameter stability bar chart
    param_stability_items = {k: v for k, v in robustness_metrics.items() 
                           if 'Stability' in k and not k.startswith('Avg') and '(0-1)' not in k}
    
    if param_stability_items:
        ax3 = plt.subplot(2, 2, 3)
        
        # Sort parameters by stability
        sorted_param_stability = dict(sorted(param_stability_items.items(), 
                                           key=lambda item: item[1], reverse=True))
        
        # Plot bar chart
        bars = plt.barh(
            list(sorted_param_stability.keys()), 
            list(sorted_param_stability.values()),
            color='lightgreen'
        )
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.01
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center')
        
        plt.xlim(0, 1.1)
        plt.xlabel('Stability Score (0-1 Scale)', fontsize=12)
        plt.ylabel('Parameter', fontsize=12)
        plt.title('Parameter Stability Analysis', fontsize=14)
        plt.grid(True, axis='x', alpha=0.3)
    
    # Create text summary of robustness analysis
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')  # Turn off axis
    
    # Prepare summary text
    summary_text = "Robustness Analysis Summary:\n\n"
    
    if 'Overall Robustness Score (0-1)' in robustness_metrics:
        score = robustness_metrics['Overall Robustness Score (0-1)']
        summary_text += f"Overall Robustness: {score:.2f}/1.00\n\n"
        
        # Add interpretation
        if score >= 0.8:
            summary_text += "Interpretation: Excellent robustness - highly consistent across all metrics.\n"
        elif score >= 0.6:
            summary_text += "Interpretation: Good robustness - consistent performance with minor variations.\n"
        elif score >= 0.4:
            summary_text += "Interpretation: Moderate robustness - some inconsistencies but generally acceptable.\n"
        elif score >= 0.2:
            summary_text += "Interpretation: Low robustness - significant inconsistencies across metrics.\n"
        else:
            summary_text += "Interpretation: Poor robustness - extremely inconsistent performance.\n"
    
    # Add details about positive periods
    if 'Positive Periods (%)' in robustness_metrics:
        pos_periods = robustness_metrics['Positive Periods (%)']
        summary_text += f"\nPositive Periods: {pos_periods:.1f}% of out-of-sample windows\n"
    
    # Add OOS/IS comparison if available
    if 'OOS/IS Consistency (0-1)' in robustness_metrics:
        oos_is = robustness_metrics['OOS/IS Consistency (0-1)']
        summary_text += f"OOS/IS Consistency: {oos_is:.2f}/1.00\n"
        
        if oos_is >= 0.8:
            summary_text += "    (Very small performance drop from in-sample to out-of-sample)\n"
        elif oos_is >= 0.5:
            summary_text += "    (Moderate performance drop from in-sample to out-of-sample)\n"
        else:
            summary_text += "    (Significant performance drop from in-sample to out-of-sample)\n"
    
    # Add parameter stability info
    if 'Avg Parameter Stability (0-1)' in robustness_metrics:
        param_stab = robustness_metrics['Avg Parameter Stability (0-1)']
        summary_text += f"\nParameter Stability: {param_stab:.2f}/1.00\n"
        
        if param_stab >= 0.8:
            summary_text += "    (Highly stable parameters across windows - strategy is robust)\n"
        elif param_stab >= 0.5:
            summary_text += "    (Moderately stable parameters - acceptable robustness)\n"
        else:
            summary_text += "    (Unstable parameters - may indicate curve-fitting)\n"
    
    # Add the text to the plot
    plt.text(0, 1.0, summary_text, fontsize=11, va='top', linespacing=1.5)
    
    # Add overall title
    plt.suptitle('Strategy Robustness Analysis Dashboard', fontsize=18, y=0.98)
    
    # Add description at the bottom
    plt.figtext(0.5, 0.01, 
               "This dashboard provides a comprehensive analysis of strategy robustness based on WFO results.\n"
               "Higher scores (closer to 1.0) indicate better robustness across all metrics.\n"
               "A truly robust strategy should perform consistently across different market conditions.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(top=0.9)
    plt.show()

def generate_wfo_report_pdf(wfo_results, df, output_path=None, strategy_name="ATDMF Strategy"):
    """
    Génère un rapport PDF complet contenant tous les résultats et graphiques
    de l'analyse Walk-Forward Optimization.
    
    Parameters:
    -----------
    wfo_results : dict
        Résultats du walk_forward_optimization
    df : pandas.DataFrame
        DataFrame OHLCV original
    output_path : str, optional
        Chemin où sauvegarder le PDF. Par défaut, "{strategy_name}_WFO_Report_{date}.pdf"
    strategy_name : str, optional
        Nom de la stratégie pour le titre du rapport
        
    Returns:
    --------
    str
        Chemin vers le fichier PDF généré
    """
    
    # Configurer le style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['figure.dpi'] = 100
    
    # Créer le chemin du fichier de sortie s'il n'est pas spécifié
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'WFO_Reports'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{strategy_name.replace(' ', '_')}_WFO_Report_{timestamp}.pdf"
    
    # Extraire les données de performance
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    else:
        oos_df = pd.DataFrame()
    
    params_df = pd.DataFrame(wfo_results['best_params'])
    
    # Créer le PDF
    with PdfPages(output_path) as pdf:
        
        # =====================================================================
        # PAGE DE TITRE
        # =====================================================================
        plt.figure(figsize=(14, 10))
        plt.axis('off')
        
        # Titre du rapport
        plt.text(0.5, 0.8, f"Rapport d'Analyse Walk-Forward Optimization", 
                fontsize=15, ha='center')
        plt.text(0.5, 0.7, f"{strategy_name}", fontsize=24, ha='center')
        
        # Date et informations de base
        today = datetime.datetime.now().strftime("%d/%m/%Y")
        plt.text(0.5, 0.6, f"Généré le {today}", fontsize=16, ha='center')
        
        # Statistiques de base
        if 'settings' in wfo_results:
            settings = wfo_results['settings']
            info_text = (
                f"Nombre de fenêtres: {settings['n_windows']}\n"
                f"Type de WFO: {'Ancrée' if settings['anchored'] else 'Non-ancrée'}\n"
                f"Proportion d'entraînement: {settings['train_size']*100:.0f}%\n"
                f"Métrique primaire: {settings['optimization_metric']}\n"
                f"Métrique secondaire: {settings['secondary_metric']}\n"
            )
            plt.text(0.5, 0.45, info_text, fontsize=14, ha='center', linespacing=1.5)
        
        # Icône ou logo (optionnel)
        # Vous pourriez ajouter un logo ici si nécessaire
        
        # Ajouter la première page
        pdf.savefig()
        plt.close()
        
        # =====================================================================
        # RÉSUMÉ DES PERFORMANCES
        # =====================================================================
        plt.figure(figsize=(14, 10))
        plt.axis('off')
        
        plt.text(0.5, 0.95, "Résumé des Performances", fontsize=24, ha='center')
        
        # Tableau des performances OOS
        if not oos_df.empty:
            # Calculer les statistiques agrégées
            summary = {
                "Rendement moyen (%)": oos_df['return'].mean() if 'return' in oos_df.columns else np.nan,
                "Rendement cumulatif (%)": ((1 + oos_df['return']/100).prod() - 1) * 100 if 'return' in oos_df.columns else np.nan, 
                "Ratio de Sharpe moyen": oos_df['sharpe'].mean() if 'sharpe' in oos_df.columns else np.nan,
                "Drawdown maximum moyen (%)": oos_df['max_drawdown'].mean() if 'max_drawdown' in oos_df.columns else np.nan,
                "Taux de réussite moyen (%)": oos_df['win_rate'].mean() if 'win_rate' in oos_df.columns else np.nan,
                "Ratio de Calmar moyen": oos_df['calmar_ratio'].mean() if 'calmar_ratio' in oos_df.columns else np.nan,
                "Ratio de Sortino moyen": oos_df['sortino_ratio'].mean() if 'sortino_ratio' in oos_df.columns else np.nan,
                "Nombre total de trades": oos_df['n_trades'].sum() if 'n_trades' in oos_df.columns else np.nan,
                "% de périodes positives": (oos_df['return'] > 0).mean() * 100 if 'return' in oos_df.columns else np.nan
            }
            
            # Créer un tableau pour les résultats résumés
            table_data = []
            for metric, value in summary.items():
                if not np.isnan(value):
                    if "(%)" in metric:
                        formatted_value = f"{value:.2f}%"
                    elif "Nombre" in metric:
                        formatted_value = f"{int(value)}"
                    else:
                        formatted_value = f"{value:.4f}"
                    table_data.append([metric, formatted_value])
            
            if table_data:
                table = plt.table(cellText=table_data, 
                                 colLabels=["Métrique", "Valeur"], 
                                 loc='center', 
                                 cellLoc='center', 
                                 bbox=[0.2, 0.6, 0.6, 0.25])
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 1.5)
                
                # Ajouter un titre au tableau
                plt.text(0.5, 0.87, "Métriques Agrégées Out-of-Sample", fontsize=16, ha='center')
        
        # Tableau des paramètres optimaux agrégés
        if not params_df.empty:
            # Calculer les paramètres optimaux agrégés (moyenne)
            aggregated_params = {}
            for param in params_df.columns:
                if pd.api.types.is_numeric_dtype(params_df[param]):
                    if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length']:
                        aggregated_params[param] = int(round(params_df[param].mean()))
                    else:
                        aggregated_params[param] = round(params_df[param].mean(), 2)
            
            # Créer un tableau pour les paramètres agrégés
            param_data = [[param, value] for param, value in aggregated_params.items()]
            
            if param_data:
                param_table = plt.table(cellText=param_data, 
                                      colLabels=["Paramètre", "Valeur Moyenne"], 
                                      loc='center', 
                                      cellLoc='center', 
                                      bbox=[0.2, 0.25, 0.6, 0.25])
                param_table.auto_set_font_size(False)
                param_table.set_fontsize(12)
                param_table.scale(1, 1.5)
                
                # Ajouter un titre au tableau
                plt.text(0.5, 0.52, "Paramètres Optimaux Agrégés", fontsize=16, ha='center')
        
        # Ajouter la page de résumé
        pdf.savefig()
        plt.close()
        
        # =====================================================================
        # GRAPHIQUE DES PRIX AVEC FENÊTRES WFO
        # =====================================================================
        
        # Fonction utilitaire pour capturer les figures matplotlib
        def save_figure_to_pdf(pdf):
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            pdf.savefig()
            plt.close()
            
        # Fonction utilitaire pour capturer les figures plotly
        def save_plotly_to_pdf(fig, pdf):
            buf = io.BytesIO()
            pio.write_image(fig, buf, format='png', width=900, height=600)
            buf.seek(0)
            img = Image.open(buf)
            
            plt.figure(figsize=(14, 10))
            plt.imshow(np.array(img))
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
        # Graphique des prix avec fenêtres WFO
        plt.figure(figsize=(14, 10))
        
        # Plot the close price
        if 'Close' in df.columns:
            price_series = df['Close']
        elif 'close' in df.columns:
            price_series = df['close']
        else:
            price_series = df.iloc[:, 0]  # En dernier recours
        
        # Tracer la série de prix
        plt.plot(df.index, price_series, label='Prix', color='#1f77b4')
        
        # Ajouter des lignes verticales pour les limites des fenêtres
        window_colors = {
            'train': 'green',
            'test': 'red'
        }
        
        # Ajouter des lignes verticales pour les limites des fenêtres
        for window_idx, window_result in enumerate(wfo_results['window_results']):
            window_info = window_result['window_info']
            
            # Ajouter une ligne verticale au début de la fenêtre
            plt.axvline(x=pd.to_datetime(window_info['start_date']), 
                       color='black', linestyle='--', alpha=0.5)
            
            # Marquer les régions in-sample et out-of-sample
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
        
        # Personnaliser le graphique
        plt.title('Graphique des Prix avec Fenêtres Walk-Forward Optimization', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Prix', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        save_figure_to_pdf(pdf)
        
        # =====================================================================
        # COMPARAISON IN-SAMPLE VS OUT-OF-SAMPLE
        # =====================================================================
        
        # Extraire les métriques de performance pour in-sample et out-of-sample
        is_metrics = []
        for window_idx, window_result in enumerate(wfo_results['window_results']):
            # Obtenir le meilleur résultat in-sample des résultats d'optimisation
            if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
                best_result = window_result['optimization_results'][0]
                if 'combined_score' in best_result:
                    is_metrics.append({
                        'window': window_idx + 1,
                        'performance': best_result['combined_score'],
                        'type': 'In-Sample'
                    })
        
        # Obtenir les métriques out-of-sample
        oos_metrics = []
        for metric in oos_df.to_dict('records'):
            oos_metrics.append({
                'window': metric['window'],
                'performance': metric['return'] if 'return' in metric else (
                              metric['sharpe'] if 'sharpe' in metric else 0),
                'type': 'Out-of-Sample'
            })
        
        # Combiner les métriques
        comparison_df = pd.DataFrame(is_metrics + oos_metrics)
        
        if not comparison_df.empty:
            plt.figure(figsize=(14, 10))
            
            # Créer un graphique à barres groupées
            ax = sns.barplot(
                x='window', 
                y='performance', 
                hue='type',
                data=comparison_df,
                palette={'In-Sample': 'skyblue', 'Out-of-Sample': 'salmon'}
            )
            
            # Ajouter des étiquettes de valeur sur les barres
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                       f'{height:.2f}', ha="center")
            
            plt.title('Comparaison In-Sample vs Out-of-Sample par Fenêtre', fontsize=16)
            plt.xlabel('Fenêtre', fontsize=14)
            plt.ylabel('Performance', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Type')
            
            save_figure_to_pdf(pdf)
        
        # =====================================================================
        # ANALYSE DE STABILITÉ DES PARAMÈTRES
        # =====================================================================
        
        # Cette visualisation montre comment les paramètres évoluent à travers les fenêtres
        
        # Filtrer uniquement les paramètres numériques
        numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_params) > 0:
            # Créer un graphique de stabilité des paramètres
            plt.figure(figsize=(14, 10))
            
            # Obtenir les valeurs de paramètres normalisées pour comparaison à travers différentes échelles
            norm_params_df = pd.DataFrame()
            for param in numeric_params:
                # Ignorer si toutes les valeurs sont identiques (stabilité = 1.0)
                if params_df[param].nunique() <= 1:
                    continue
                    
                # Normaliser à l'échelle 0-1 pour tracer sur la même échelle
                min_val = params_df[param].min()
                max_val = params_df[param].max()
                
                if max_val > min_val:  # Éviter la division par zéro
                    norm_params_df[param] = (params_df[param] - min_val) / (max_val - min_val)
                else:
                    norm_params_df[param] = params_df[param] / params_df[param]
            
            # Créer l'axe des x (numéros de fenêtre)
            x = list(range(1, len(params_df) + 1))
            
            # Tracer chaque paramètre
            markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', '|', '_']
            for i, param in enumerate(norm_params_df.columns):
                marker = markers[i % len(markers)]
                plt.plot(x, norm_params_df[param], marker=marker, label=param, linewidth=2, markersize=8)
                
                # Calculer la métrique de stabilité (1 - coefficient de variation)
                stability = 1.0 - (np.std(params_df[param]) / np.mean(params_df[param])) if np.mean(params_df[param]) > 0 else 0.0
                
                # Annoter avec la valeur de stabilité
                plt.annotate(f"Stabilité: {stability:.2f}", 
                            xy=(x[-1], norm_params_df[param].iloc[-1]),
                            xytext=(x[-1] + 0.1, norm_params_df[param].iloc[-1]),
                            fontsize=9)
            
            # Ajouter un tableau de valeurs absolues des paramètres
            param_table = ''
            for i, window in enumerate(x):
                param_table += f'Fenêtre {window}: '
                param_values = []
                for param in numeric_params:
                    if param in norm_params_df.columns:
                        param_values.append(f"{param}={params_df[param].iloc[i]:.2f}")
                param_table += ', '.join(param_values) + '\n'
            
            plt.figtext(0.5, 0.01, param_table, ha="center", fontsize=9, 
                       bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            # Personnaliser le graphique
            plt.title('Stabilité des Paramètres à Travers les Fenêtres (Valeurs Normalisées)', fontsize=16)
            plt.xlabel('Fenêtre', fontsize=14)
            plt.ylabel('Valeur Normalisée du Paramètre', fontsize=14)
            plt.xticks(x)
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.05, 1.05)  # Donner une marge au-dessus et en-dessous
            plt.legend(loc='best')
            
            save_figure_to_pdf(pdf)
        
        # =====================================================================
        # DASHBOARD DES MÉTRIQUES DE PERFORMANCE OUT-OF-SAMPLE
        # =====================================================================
        
        if not oos_df.empty:
            # Créer des graphiques séparés pour chaque métrique clé
            
            # 1. Graphique des rendements
            if 'return' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                window_nums = list(range(1, len(oos_df) + 1))
                
                bars = plt.bar(window_nums, oos_df['return'], color='royalblue')
                
                # Ajouter des étiquettes de valeur sur les barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}%', ha='center', va='bottom')
                
                # Ajouter une ligne de référence pour la moyenne
                plt.axhline(y=oos_df['return'].mean(), color='red', linestyle='--', 
                          label=f'Moyenne: {oos_df["return"].mean():.2f}%')
                
                plt.title('Rendements par Fenêtre (%)', fontsize=16)
                plt.xlabel('Fenêtre', fontsize=14)
                plt.ylabel('Rendement (%)', fontsize=14)
                plt.xticks(window_nums)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_figure_to_pdf(pdf)
            
            # 2. Graphique du ratio de Sharpe
            if 'sharpe' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                
                bars = plt.bar(window_nums, oos_df['sharpe'], color='green')
                
                # Ajouter des étiquettes de valeur sur les barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
                
                # Ajouter une ligne de référence pour la moyenne
                plt.axhline(y=oos_df['sharpe'].mean(), color='red', linestyle='--', 
                          label=f'Moyenne: {oos_df["sharpe"].mean():.2f}')
                
                plt.title('Ratio de Sharpe par Fenêtre', fontsize=16)
                plt.xlabel('Fenêtre', fontsize=14)
                plt.ylabel('Ratio de Sharpe', fontsize=14)
                plt.xticks(window_nums)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_figure_to_pdf(pdf)
            
            # 3. Graphique du Drawdown Maximum
            if 'max_drawdown' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                
                bars = plt.bar(window_nums, oos_df['max_drawdown'], color='firebrick')
                
                # Ajouter des étiquettes de valeur sur les barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}%', ha='center', va='bottom')
                
                # Ajouter une ligne de référence pour la moyenne
                plt.axhline(y=oos_df['max_drawdown'].mean(), color='black', linestyle='--', 
                          label=f'Moyenne: {oos_df["max_drawdown"].mean():.2f}%')
                
                plt.title('Drawdown Maximum par Fenêtre (%)', fontsize=16)
                plt.xlabel('Fenêtre', fontsize=14)
                plt.ylabel('Drawdown Maximum (%)', fontsize=14)
                plt.xticks(window_nums)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_figure_to_pdf(pdf)
            
            # 4. Graphique du Taux de Réussite et Nombre de Trades
            if 'win_rate' in oos_df.columns and 'n_trades' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                
                fig, ax1 = plt.subplots(figsize=(14, 10))
                
                # Tracer le nombre de trades comme des barres
                bars = ax1.bar(np.array(window_nums) - 0.2, oos_df['n_trades'], 
                             width=0.4, color='skyblue', label='Nombre de Trades')
                ax1.set_xlabel('Fenêtre', fontsize=14)
                ax1.set_ylabel('Nombre de Trades', fontsize=14, color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Ajouter les étiquettes du nombre de trades au-dessus des barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points de décalage vertical
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
                
                # Tracer le taux de réussite sur le même graphique avec un axe y secondaire
                ax2 = ax1.twinx()
                bars2 = ax2.bar(np.array(window_nums) + 0.2, oos_df['win_rate'], 
                              width=0.4, color='salmon', label='Taux de Réussite (%)')
                ax2.set_ylabel('Taux de Réussite (%)', fontsize=14, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Ajouter les étiquettes du taux de réussite au-dessus des barres
                for i, bar in enumerate(bars2):
                    height = bar.get_height()
                    ax2.annotate(f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points de décalage vertical
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
                
                # Ajouter une ligne horizontale pour le taux de réussite moyen
                avg_win_rate = oos_df['win_rate'].mean()
                ax2.axhline(y=avg_win_rate, color='red', linestyle='dashed', alpha=0.8, 
                           label=f'Taux moyen: {avg_win_rate:.2f}%')
                
                # Définir les ticks x aux positions des barres
                ax1.set_xticks(window_nums)
                
                # Titre et grille
                plt.title('Métriques de Trading par Fenêtre', fontsize=16)
                ax1.grid(True, axis='y', alpha=0.3)
                
                # Créer une légende combinée
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                save_figure_to_pdf(pdf)
        
        # =====================================================================
        # ANALYSE DES IMPACTS DES PARAMÈTRES
        # =====================================================================
        
        # Cette visualisation montre comment différents paramètres impactent les métriques de performance
        
        if len(numeric_params) >= 2 and len(oos_df) >= 3:
            # Calculer la corrélation entre les paramètres et les métriques
            correlation_data = []
            
            for param in numeric_params:
                for metric in ['return', 'sharpe', 'max_drawdown', 'win_rate']:
                    if metric in oos_df.columns and not oos_df[metric].isna().all():
                        # Ignorer si toutes les valeurs de paramètres sont identiques
                        if params_df[param].nunique() <= 1:
                            continue
                            
                        # Obtenir des valeurs valides pour le calcul de corrélation
                        param_values = params_df[param].values
                        metric_values = oos_df[metric].values
                        valid_mask = ~np.isnan(param_values) & ~np.isnan(metric_values)
                        
                        if sum(valid_mask) >= 3:  # Besoin d'au moins 3 points valides pour une corrélation significative
                            try:
                                corr = np.corrcoef(param_values[valid_mask], metric_values[valid_mask])[0, 1]
                                if not np.isnan(corr) and not np.isinf(corr):
                                    correlation_data.append({
                                        'Paramètre': param,
                                        'Métrique': metric,
                                        'Corrélation': corr
                                    })
                            except Exception as e:
                                print(f"Impossible de calculer la corrélation pour {param} vs {metric}: {e}")
            
            # Créer une heatmap de corrélation si nous avons des données
            if correlation_data:
                corr_df = pd.DataFrame(correlation_data)
                # Pivoter pour créer une matrice adaptée à la heatmap
                pivot_df = corr_df.pivot(index='Paramètre', columns='Métrique', values='Corrélation')
                
                plt.figure(figsize=(14, 10))
                heatmap = sns.heatmap(
                    pivot_df,
                    cmap="coolwarm",
                    annot=True,
                    fmt=".2f",
                    linewidths=0.5,
                    center=0,
                    vmin=-1,
                    vmax=1,
                    cbar_kws={"shrink": .8, "label": "Coefficient de Corrélation"}
                )
                
                plt.title('Impact des Paramètres sur les Métriques de Performance', fontsize=16)
                plt.ylabel('Paramètre', fontsize=14)
                plt.xlabel('Métrique de Performance', fontsize=14)
                
                save_figure_to_pdf(pdf)
        
        # =====================================================================
        # DISTRIBUTION DES RENDEMENTS (ANALYSE DE RISQUE)
        # =====================================================================
        
        # Cette visualisation montre la distribution des rendements et les métriques de risque clés
        if 'return' in oos_df.columns and not oos_df['return'].isna().all():
            plt.figure(figsize=(14, 10))
            
            # Créer un graphique de distribution avec estimation de densité par noyau
            sns.histplot(oos_df['return'].values, kde=True, stat="density", 
                       color='skyblue', bins=min(10, len(oos_df)))
            
            # Marquer les statistiques importantes
            mean_return = oos_df['return'].mean()
            median_return = oos_df['return'].median()
            std_return = oos_df['return'].std()
            
            # Ajouter des lignes verticales pour la moyenne, la médiane
            plt.axvline(mean_return, color='red', linestyle='dashed', linewidth=2, 
                      label=f'Moyenne: {mean_return:.2f}%')
            plt.axvline(median_return, color='green', linestyle='dashed', linewidth=2, 
                       label=f'Médiane: {median_return:.2f}%')
            
            # Ajouter des lignes verticales pour moyenne ± 1 écart-type (intervalle de confiance 68%)
            plt.axvline(mean_return + std_return, color='purple', linestyle='dotted', 
                      label=f'Moyenne + 1σ: {mean_return + std_return:.2f}%')
            plt.axvline(mean_return - std_return, color='purple', linestyle='dotted', 
                      label=f'Moyenne - 1σ: {mean_return - std_return:.2f}%')
            
            # Ajouter une ligne horizontale à y=0 pour souligner les rendements négatifs
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Calculer les métriques clés pour le texte
            negative_returns = (oos_df['return'] < 0).mean() * 100
            positive_returns = (oos_df['return'] > 0).mean() * 100
            skewness = oos_df['return'].skew()
            kurtosis = oos_df['return'].kurtosis()
            
            # Ajouter une zone de texte avec les métriques
            metrics_text = (
                f"Métriques de Distribution:\n"
                f"Rendement Moyen: {mean_return:.2f}%\n"
                f"Rendement Médian: {median_return:.2f}%\n"
                f"Écart-Type: {std_return:.2f}%\n"
                f"Fenêtres Négatives: {negative_returns:.1f}%\n"
                f"Fenêtres Positives: {positive_returns:.1f}%\n"
                f"Asymétrie: {skewness:.2f}\n"
                f"Kurtosis: {kurtosis:.2f}"
            )
            
            # Ajouter la zone de texte des métriques
            plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        backgroundcolor='white', alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                        verticalalignment='top')
            
            # Personnaliser le graphique
            plt.title('Distribution des Rendements Out-of-Sample', fontsize=16)
            plt.xlabel('Rendement (%)', fontsize=14)
            plt.ylabel('Densité', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            save_figure_to_pdf(pdf)
            
        # =====================================================================
        # ANALYSE DE ROBUSTESSE DE LA STRATÉGIE
        # =====================================================================
        
        # Calculer les métriques de robustesse
        robustness_metrics = {}
        
        # 1. Cohérence de Performance: % de périodes OOS positives
        if 'return' in oos_df.columns:
            positive_periods = (oos_df['return'] > 0).mean() * 100
            robustness_metrics['Périodes Positives (%)'] = positive_periods
        
        # 2. Stabilité de Performance: coefficient de variation des rendements
        if 'return' in oos_df.columns and not oos_df['return'].isna().all():
            # Calculer coefficient de variation (std/mean) pour non-zero mean
            mean_return = oos_df['return'].mean()
            if abs(mean_return) > 1e-6:  # Éviter division par zéro ou nombres minuscules
                cv_return = oos_df['return'].std() / abs(mean_return)
                # Convertir en stabilité (1 - CV normalisé)
                # Limiter à la plage [0, 1] en utilisant 1 / (1 + CV)
                return_stability = 1 / (1 + cv_return)
                robustness_metrics['Stabilité des Rendements (0-1)'] = return_stability
        
        # 3. Cohérence de Sharpe: coefficient de variation des ratios de Sharpe
        if 'sharpe' in oos_df.columns and not oos_df['sharpe'].isna().all():
            # Seulement pour la moyenne de Sharpe positive
            mean_sharpe = oos_df['sharpe'].mean()
            if mean_sharpe > 1e-6:
                cv_sharpe = oos_df['sharpe'].std() / mean_sharpe
                sharpe_stability = 1 / (1 + cv_sharpe)
                robustness_metrics['Stabilité de Sharpe (0-1)'] = sharpe_stability
        
        # 4. Cohérence du Taux de Réussite
        if 'win_rate' in oos_df.columns and not oos_df['win_rate'].isna().all():
            mean_win_rate = oos_df['win_rate'].mean()
            if mean_win_rate > 1e-6:
                cv_win_rate = oos_df['win_rate'].std() / mean_win_rate
                win_rate_stability = 1 / (1 + cv_win_rate)
                robustness_metrics['Stabilité du Taux de Réussite (0-1)'] = win_rate_stability
        
        # 5. Stabilité des Paramètres: moyenne de 1 - CV pour chaque paramètre
        param_stability = {}
        numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
        
        for param in numeric_params:
            mean_value = params_df[param].mean()
            if abs(mean_value) > 1e-6:  # Éviter division par zéro
                cv = params_df[param].std() / abs(mean_value)
                stability = 1 / (1 + cv)
                param_stability[param] = stability
        
        if param_stability:
            robustness_metrics['Stabilité Moy. des Paramètres (0-1)'] = np.mean(list(param_stability.values()))
            # Stocker également la stabilité individuelle des paramètres
            for param, stability in param_stability.items():
                robustness_metrics[f'Stabilité {param}'] = stability
        
        # 6. Ratio Performance OOS vs IS 
        # Si nous avons les métriques IS et OOS, calculer le ratio
        is_metrics = []
        for window_result in wfo_results['window_results']:
            if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
                best_result = window_result['optimization_results'][0]
                if 'combined_score' in best_result:
                    is_metrics.append(best_result['combined_score'])
        
        if is_metrics and 'return' in oos_df.columns:
            is_avg = np.mean(is_metrics)
            oos_avg = oos_df['return'].mean()
            
            if abs(is_avg) > 1e-6:  # Éviter division par zéro
                oos_is_ratio = oos_avg / is_avg
                # Normaliser à l'échelle 0-1: 1 signifie OOS = IS (parfait), 0 signifie complètement différent
                oos_is_consistency = 1 - min(1, abs(1 - oos_is_ratio))
                robustness_metrics['Cohérence OOS/IS (0-1)'] = oos_is_consistency
        
        # 7. Calculer le score de robustesse global (moyenne de toutes les métriques)
        # Filtrer pour inclure uniquement les métriques à l'échelle 0-1
        scaled_metrics = {k: v for k, v in robustness_metrics.items() if '(0-1)' in k}
        if scaled_metrics:
            robustness_metrics['Score de Robustesse Global (0-1)'] = np.mean(list(scaled_metrics.values()))
        
        # Créer une page résumant les métriques de robustesse
        if robustness_metrics:
            plt.figure(figsize=(14, 10))
            plt.axis('off')
            
            plt.text(0.5, 0.95, "Analyse de Robustesse de la Stratégie", fontsize=14, ha='center')
            
            # Filtrer les métriques pour le graphique radar
            radar_metrics = {k.replace(' (0-1)', ''): v for k, v in robustness_metrics.items() if '(0-1)' in k}
            
            # Si nous avons suffisamment de métriques pour un graphique radar
            if len(radar_metrics) >= 3:
                # Créer un graphique radar (spider plot) pour les métriques de robustesse
                ax1 = plt.subplot2grid((2, 2), (0, 0), polar=True)
                
                # Préparer les données pour le graphique radar
                categories = list(radar_metrics.keys())
                values = list(radar_metrics.values())
                
                # Fermer le tracé en ajoutant la première valeur à la fin
                categories = categories + [categories[0]]
                values = values + [values[0]]
                
                # Calculer l'angle pour chaque catégorie
                N = len(categories) - 1  # Excluant le premier élément répété
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Fermer la boucle
                
                # Tracer le graphique
                ax1.plot(angles, values, marker='o', linestyle='-', linewidth=2, label='Métriques de Robustesse')
                
                # Fill the area
                ax1.fill(angles, values, alpha=0.25)
                
                # Définir les étiquettes de catégorie
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(categories[:-1])
                
                # Définir les limites radiales
                ax1.set_ylim(0, 1)
                
                # Ajouter des lignes de grille radiales à 0.2, 0.4, 0.6, 0.8
                plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
                
                # Dessiner des cercles d'axe y
                for ytick in [0.2, 0.4, 0.6, 0.8]:
                    ax1.add_artist(plt.Circle((0, 0), ytick, fill=False, color='grey', linestyle='--', alpha=0.4))
                
                # Titre pour ce sous-graphique
                ax1.set_title('Métriques de Robustesse de la Stratégie (1.0 = Idéal)', fontsize=14, pad=20)
            
            # Créer un graphique à barres avec toutes les métriques de robustesse
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            
            metrics_to_plot = {k: v for k, v in robustness_metrics.items() 
                             if not k.startswith('Stabilité Moy.') and '(0-1)' in k}
            
            # Sort metrics by value
            sorted_metrics = dict(sorted(metrics_to_plot.items(), key=lambda item: item[1], reverse=True))
            
            # Plot bar chart
            bars = plt.barh(
                [k.replace(' (0-1)', '') for k in sorted_metrics.keys()], 
                list(sorted_metrics.values()),
                color='skyblue'
            )
            
            # Ajouter des étiquettes de valeur
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width + 0.01
                plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
               f'{width:.2f}', va='center')
            
            plt.xlim(0, 1.1)
            plt.xlabel('Score (échelle 0-1)', fontsize=12)
            plt.ylabel('Métrique', fontsize=12)
            plt.title('Comparaison des Métriques de Robustesse', fontsize=14)
            plt.grid(True, axis='x', alpha=0.3)
            
            # Créer un graphique à barres de stabilité des paramètres
            param_stability_items = {k: v for k, v in robustness_metrics.items() 
                                   if 'Stabilité' in k and not k.startswith('Stabilité Moy.') and '(0-1)' not in k}
            
            if param_stability_items:
                ax3 = plt.subplot2grid((2, 2), (1, 0))
                
                # Trier les paramètres par stabilité
                sorted_param_stability = dict(sorted(param_stability_items.items(), 
                                                   key=lambda item: item[1], reverse=True))
                
                # Tracer un graphique à barres
                bars = plt.barh(
                    list(sorted_param_stability.keys()), 
                    list(sorted_param_stability.values()),
                    color='lightgreen'
                )
                
                # Ajouter des étiquettes de valeur
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    label_x_pos = width + 0.01
                    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center')
                
                plt.xlim(0, 1.1)
                plt.xlabel('Score de Stabilité (échelle 0-1)', fontsize=12)
                plt.ylabel('Paramètre', fontsize=12)
                plt.title('Analyse de Stabilité des Paramètres', fontsize=14)
                plt.grid(True, axis='x', alpha=0.3)
            
            # Créer un résumé textuel de l'analyse de robustesse
            ax4 = plt.subplot2grid((2, 2), (1, 1))
            ax4.axis('off')  # Désactiver l'axe
            
            # Préparer le texte de résumé
            summary_text = "Résumé de l'Analyse de Robustesse:\n\n"
            
            if 'Score de Robustesse Global (0-1)' in robustness_metrics:
                score = robustness_metrics['Score de Robustesse Global (0-1)']
                summary_text += f"Robustesse Globale: {score:.2f}/1.00\n\n"
                
                # Ajouter une interprétation
                if score >= 0.8:
                    summary_text += "Interprétation: Excellente robustesse - très cohérente sur toutes les métriques.\n"
                elif score >= 0.6:
                    summary_text += "Interprétation: Bonne robustesse - performance cohérente avec des variations mineures.\n"
                elif score >= 0.4:
                    summary_text += "Interprétation: Robustesse modérée - quelques incohérences mais généralement acceptable.\n"
                elif score >= 0.2:
                    summary_text += "Interprétation: Faible robustesse - incohérences significatives entre les métriques.\n"
                else:
                    summary_text += "Interprétation: Mauvaise robustesse - performance extrêmement incohérente.\n"
            
            # Ajouter des détails sur les périodes positives
            if 'Périodes Positives (%)' in robustness_metrics:
                pos_periods = robustness_metrics['Périodes Positives (%)']
                summary_text += f"\nPériodes positives: {pos_periods:.1f}% des fenêtres out-of-sample\n"
            
            # Ajouter une comparaison OOS/IS si disponible
            if 'Cohérence OOS/IS (0-1)' in robustness_metrics:
                oos_is = robustness_metrics['Cohérence OOS/IS (0-1)']
                summary_text += f"Cohérence OOS/IS: {oos_is:.2f}/1.00\n"
                
                if oos_is >= 0.8:
                    summary_text += "    (Très faible baisse de performance de in-sample à out-of-sample)\n"
                elif oos_is >= 0.5:
                    summary_text += "    (Baisse de performance modérée de in-sample à out-of-sample)\n"
                else:
                    summary_text += "    (Baisse de performance significative de in-sample à out-of-sample)\n"
            
            # Add parameter stability info
            if 'Stabilité Moy. des Paramètres (0-1)' in robustness_metrics:
                param_stab = robustness_metrics['Stabilité Moy. des Paramètres (0-1)']
                summary_text += f"\nStabilité des Paramètres: {param_stab:.2f}/1.00\n"
                
                if param_stab >= 0.8:
                    summary_text += "    (Paramètres très stables à travers les fenêtres - stratégie robuste)\n"
                elif param_stab >= 0.5:
                    summary_text += "    (Paramètres modérément stables - robustesse acceptable)\n"
                else:
                    summary_text += "    (Paramètres instables - peut indiquer un surajustement)\n"
            
            # Ajouter le texte au graphique
            ax4.text(0, 1.0, summary_text, fontsize=11, va='top', linespacing=1.5)
            
            # Ajouter un titre général
            plt.suptitle('Strategy Robustness Analysis Dashboard', fontsize=18, y=0.98)
            
            # Ajouter une description en bas
            plt.figtext(0.5, 0.01, 
               "This dashboard provides a comprehensive analysis of strategy robustness based on WFO results.\n"
               "Higher scores (closer to 1.0) indicate better robustness across all metrics.\n"
               "A truly robust strategy should perform consistently across different market conditions.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.subplots_adjust(top=0.9)
            plt.show()

def integrate_report_generation(wfo_results, df, strategy_name="ATDMF Strategy"):
    """
    Fonction wrapper pour intégrer la génération de rapport dans le workflow principal.
    
    Parameters:
    -----------
    wfo_results : dict
        Résultats du walk_forward_optimization
    df : pandas.DataFrame
        DataFrame OHLCV original
    strategy_name : str, optional
        Nom de la stratégie pour le titre du rapport
        
    Returns:
    --------
    None
    """
    # Demander à l'utilisateur s'il souhaite générer un rapport PDF
    generate_report = input("\nGénérer un rapport PDF complet? (o/n) [default: o]: ").lower()
    
    if generate_report != 'n':
        print("\nGénération du rapport PDF en cours...")
        try:
            report_path = generate_wfo_report_pdf(wfo_results, df, strategy_name=strategy_name)
            print(f"Rapport généré avec succès: {report_path}")
            
            # Option pour ouvrir automatiquement le PDF
            open_report = input("Ouvrir le rapport PDF maintenant? (o/n) [default: o]: ").lower()
            if open_report != 'n':
                
                if platform.system() == 'Darwin':  # macOS
                    subprocess.call(('open', report_path))
                elif platform.system() == 'Windows':  # Windows
                    os.startfile(report_path)
                else:  # Linux
                    subprocess.call(('xdg-open', report_path))
                    
        except Exception as e:
            print(f"Erreur lors de la génération du rapport PDF: {e}")
            print("Vérifiez que vous avez installé toutes les dépendances nécessaires:")
            print("  - matplotlib")
            print("  - pandas")
            print("  - numpy")
            print("  - seaborn")
            print("  - plotly")
            print("  - Pillow (PIL)")
    else:
        print("Génération du rapport PDF ignorée.")
