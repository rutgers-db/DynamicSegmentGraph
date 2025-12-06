#!/usr/bin/env python3
"""
Deep KNN-First Performance Visualization: Filtered vs Unfiltered Comparison

This script analyzes and visualizes the performance metrics (recall and QPS) 
from Deep_KNNFirst_M*.log files, comparing filtered and unfiltered top candidate approaches.
The visualization shows recall-QPS trade-off curves for different ranges and M values, 
where each range has its own subplot with two curves: filtered vs unfiltered approaches.

Author: Generated for HNSW performance analysis
Created: 2024
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def parse_log_file(filepath):
    """
    Parse the Deep_KNNFirst_M*.log file and extract performance metrics.
    
    Args:
        filepath (str): Path to the log file
        
    Returns:
        dict: Nested dictionary with structure {ef_value: {range_value: {'recall': float, 'qps': float}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Pattern to match search ef sections
    ef_pattern = r'Search ef: (\d+)\n=+\n(.*?)\n=+'
    
    for ef_match in re.finditer(ef_pattern, content, re.DOTALL):
        ef_value = int(ef_match.group(1))
        section_content = ef_match.group(2)
        
        # Pattern to match range, recall, and QPS lines
        line_pattern = r'range: (\d+)\s+recall: ([\d.]+)\s+QPS: (\d+)'
        
        for line_match in re.finditer(line_pattern, section_content):
            range_value = int(line_match.group(1))
            recall = float(line_match.group(2))
            qps = int(line_match.group(3))
            
            data[ef_value][range_value] = {'recall': recall, 'qps': qps}
    
    return data

def create_recall_qps_tradeoff_plots(filtered_data, unfiltered_data, m_value, output_dir):
    """
    Create recall-QPS trade-off plots with one subplot per range for a specific M value.
    
    Args:
        filtered_data (dict): Parsed data from filtered log file
        unfiltered_data (dict): Parsed data from unfiltered log file
        m_value (str): The M value (e.g., 'M16', 'M32', 'M64')
        output_dir (str): Directory to save the plots
    """
    # Set up matplotlib parameters for better plots
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    
    # Get all unique ranges from both datasets
    all_ranges = set()
    for ef_data in filtered_data.values():
        all_ranges.update(ef_data.keys())
    for ef_data in unfiltered_data.values():
        all_ranges.update(ef_data.keys())
    all_ranges = sorted(list(all_ranges))
    
    # Create subplots - arrange in a grid
    n_ranges = len(all_ranges)
    n_cols = 3
    n_rows = (n_ranges + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    fig.suptitle(f'Recall-QPS Trade-off Curves: Filtered vs Unfiltered ({m_value})', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Colors for filtered vs unfiltered
    filtered_color = '#2E86AB'  # Blue
    unfiltered_color = '#A23B72'  # Red/Purple
    
    for i, range_val in enumerate(all_ranges):
        ax = axes[i]
        
        # Extract recall-QPS points for filtered data at this range
        filtered_recalls = []
        filtered_qps = []
        filtered_ef_values = []
        
        for ef in sorted(filtered_data.keys()):
            if range_val in filtered_data[ef]:
                recall = filtered_data[ef][range_val]['recall']
                qps = filtered_data[ef][range_val]['qps']
                # Only include points with recall >= 0.8
                if recall >= 0.8:
                    filtered_recalls.append(recall)
                    filtered_qps.append(qps)
                    filtered_ef_values.append(ef)
        
        # Extract recall-QPS points for unfiltered data at this range
        unfiltered_recalls = []
        unfiltered_qps = []
        unfiltered_ef_values = []
        
        for ef in sorted(unfiltered_data.keys()):
            if range_val in unfiltered_data[ef]:
                recall = unfiltered_data[ef][range_val]['recall']
                qps = unfiltered_data[ef][range_val]['qps']
                # Only include points with recall >= 0.8
                if recall >= 0.8:
                    unfiltered_recalls.append(recall)
                    unfiltered_qps.append(qps)
                    unfiltered_ef_values.append(ef)
        
        # Plot the curves
        if filtered_recalls and filtered_qps:
            ax.plot(filtered_recalls, filtered_qps, 'o-', 
                   color=filtered_color, label='Filtered', alpha=0.8, linewidth=2)
        
        if unfiltered_recalls and unfiltered_qps:
            ax.plot(unfiltered_recalls, unfiltered_qps, 's-', 
                   color=unfiltered_color, label='Unfiltered', alpha=0.8, linewidth=2)
        
        # Customize subplot
        ax.set_title(f'Range: {range_val}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Recall')
        ax.set_ylabel('QPS (Queries Per Second)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0.8, 1.0)  # Focus on high-recall region
        
        # Add ef value annotations for key points
        if filtered_recalls and filtered_qps:
            # Annotate a few key ef values for filtered
            key_indices = [0, len(filtered_ef_values)//2, -1]  # First, middle, last
            for idx in key_indices:
                if 0 <= idx < len(filtered_ef_values):
                    ax.annotate(f'ef={filtered_ef_values[idx]}', 
                               (filtered_recalls[idx], filtered_qps[idx]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7, color=filtered_color)
    
    # Hide empty subplots
    for i in range(len(all_ranges), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'recall_qps_tradeoff_by_range_{m_value}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_comparison(filtered_data, unfiltered_data, m_value, output_dir):
    """
    Create a summary plot showing the best trade-off points for each range for a specific M value.
    
    Args:
        filtered_data (dict): Parsed data from filtered log file
        unfiltered_data (dict): Parsed data from unfiltered log file
        m_value (str): The M value (e.g., 'M16', 'M32', 'M64')
        output_dir (str): Directory to save the plots
    """
    # Get all unique ranges
    all_ranges = set()
    for ef_data in filtered_data.values():
        all_ranges.update(ef_data.keys())
    for ef_data in unfiltered_data.values():
        all_ranges.update(ef_data.keys())
    all_ranges = sorted(list(all_ranges))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Summary: Best Performance Points by Range ({m_value})', fontsize=16, fontweight='bold')
    
    # Colors for different ranges
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_ranges)))
    
    # Plot 1: All curves together
    ax1.set_title('All Recall-QPS Curves', fontweight='bold')
    for i, range_val in enumerate(all_ranges):
        # Filtered data
        filtered_recalls = []
        filtered_qps = []
        for ef in sorted(filtered_data.keys()):
            if range_val in filtered_data[ef]:
                recall = filtered_data[ef][range_val]['recall']
                qps = filtered_data[ef][range_val]['qps']
                # Only include points with recall >= 0.8
                if recall >= 0.8:
                    filtered_recalls.append(recall)
                    filtered_qps.append(qps)
        
        if filtered_recalls:
            ax1.plot(filtered_recalls, filtered_qps, 'o-', 
                    color=colors[i], label=f'Filtered Range={range_val}', alpha=0.7)
        
        # Unfiltered data
        unfiltered_recalls = []
        unfiltered_qps = []
        for ef in sorted(unfiltered_data.keys()):
            if range_val in unfiltered_data[ef]:
                recall = unfiltered_data[ef][range_val]['recall']
                qps = unfiltered_data[ef][range_val]['qps']
                # Only include points with recall >= 0.8
                if recall >= 0.8:
                    unfiltered_recalls.append(recall)
                    unfiltered_qps.append(qps)
        
        if unfiltered_recalls:
            ax1.plot(unfiltered_recalls, unfiltered_qps, 's--', 
                    color=colors[i], label=f'Unfiltered Range={range_val}', alpha=0.7)
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('QPS (Queries Per Second)')
    ax1.set_yscale('log')
    ax1.set_xlim(0.8, 1.0)  # Focus on high-recall region
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Best points comparison
    ax2.set_title('Best Trade-off Points Comparison', fontweight='bold')
    
    filtered_best_recall = []
    filtered_best_qps = []
    unfiltered_best_recall = []
    unfiltered_best_qps = []
    range_labels = []
    
    for range_val in all_ranges:
        # Find best balance point for filtered (highest recall * QPS product)
        best_filtered_score = 0
        best_filtered_recall = 0
        best_filtered_qps = 0
        
        for ef in filtered_data.keys():
            if range_val in filtered_data[ef]:
                recall = filtered_data[ef][range_val]['recall']
                qps = filtered_data[ef][range_val]['qps']
                # Only consider points with recall >= 0.8
                if recall >= 0.8:
                    score = recall * np.log(qps)  # Use log to balance the scale
                    if score > best_filtered_score:
                        best_filtered_score = score
                        best_filtered_recall = recall
                        best_filtered_qps = qps
        
        # Find best balance point for unfiltered
        best_unfiltered_score = 0
        best_unfiltered_recall = 0
        best_unfiltered_qps = 0
        
        for ef in unfiltered_data.keys():
            if range_val in unfiltered_data[ef]:
                recall = unfiltered_data[ef][range_val]['recall']
                qps = unfiltered_data[ef][range_val]['qps']
                # Only consider points with recall >= 0.8
                if recall >= 0.8:
                    score = recall * np.log(qps)
                    if score > best_unfiltered_score:
                        best_unfiltered_score = score
                        best_unfiltered_recall = recall
                        best_unfiltered_qps = qps
        
        if best_filtered_recall > 0:
            filtered_best_recall.append(best_filtered_recall)
            filtered_best_qps.append(best_filtered_qps)
            range_labels.append(range_val)
        
        if best_unfiltered_recall > 0:
            unfiltered_best_recall.append(best_unfiltered_recall)
            unfiltered_best_qps.append(best_unfiltered_qps)
    
    if filtered_best_recall:
        ax2.scatter(filtered_best_recall, filtered_best_qps, 
                   c='#2E86AB', s=100, alpha=0.8, label='Filtered Best', marker='o')
    
    if unfiltered_best_recall:
        ax2.scatter(unfiltered_best_recall, unfiltered_best_qps, 
                   c='#A23B72', s=100, alpha=0.8, label='Unfiltered Best', marker='s')
    
    # Add range labels
    for i, range_val in enumerate(range_labels):
        if i < len(filtered_best_recall):
            ax2.annotate(f'{range_val}', 
                        (filtered_best_recall[i], filtered_best_qps[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        if i < len(unfiltered_best_recall):
            ax2.annotate(f'{range_val}', 
                        (unfiltered_best_recall[i], unfiltered_best_qps[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('QPS (Queries Per Second)')
    ax2.set_yscale('log')
    ax2.set_xlim(0.8, 1.0)  # Focus on high-recall region
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_comparison_{m_value}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(filtered_data, unfiltered_data, m_value, output_dir):
    """
    Generate summary statistics comparing filtered vs unfiltered performance for a specific M value.
    
    Args:
        filtered_data (dict): Parsed data from filtered log file
        unfiltered_data (dict): Parsed data from unfiltered log file
        m_value (str): The M value (e.g., 'M16', 'M32', 'M64')
        output_dir (str): Directory to save the summary
    """
    summary = []
    summary.append(f"RECALL-QPS TRADE-OFF ANALYSIS SUMMARY ({m_value})")
    summary.append("=" * 60)
    summary.append("")
    
    # Get all ranges
    all_ranges = set()
    for ef_data in filtered_data.values():
        all_ranges.update(ef_data.keys())
    for ef_data in unfiltered_data.values():
        all_ranges.update(ef_data.keys())
    all_ranges = sorted(list(all_ranges))
    
    summary.append("Range-wise Performance Analysis:")
    summary.append("-" * 40)
    
    for range_val in all_ranges:
        summary.append(f"\nRange: {range_val}")
        summary.append("-" * 20)
        
        # Analyze filtered performance
        filtered_points = []
        for ef in sorted(filtered_data.keys()):
            if range_val in filtered_data[ef]:
                recall = filtered_data[ef][range_val]['recall']
                qps = filtered_data[ef][range_val]['qps']
                filtered_points.append((ef, recall, qps))
        
        # Analyze unfiltered performance
        unfiltered_points = []
        for ef in sorted(unfiltered_data.keys()):
            if range_val in unfiltered_data[ef]:
                recall = unfiltered_data[ef][range_val]['recall']
                qps = unfiltered_data[ef][range_val]['qps']
                unfiltered_points.append((ef, recall, qps))
        
        if filtered_points:
            summary.append("Filtered approach:")
            summary.append(f"  Recall range: {min(p[1] for p in filtered_points):.3f} - {max(p[1] for p in filtered_points):.3f}")
            summary.append(f"  QPS range: {min(p[2] for p in filtered_points)} - {max(p[2] for p in filtered_points)}")
        
        if unfiltered_points:
            summary.append("Unfiltered approach:")
            summary.append(f"  Recall range: {min(p[1] for p in unfiltered_points):.3f} - {max(p[1] for p in unfiltered_points):.3f}")
            summary.append(f"  QPS range: {min(p[2] for p in unfiltered_points)} - {max(p[2] for p in unfiltered_points)}")
    
    # Save summary to file
    with open(os.path.join(output_dir, f'tradeoff_analysis_summary_{m_value}.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    print('\n'.join(summary))

def process_m_value(m_value, base_dir):
    """
    Process a single M value and generate all visualizations.
    
    Args:
        m_value (str): The M value (e.g., 'M16', 'M32', 'M64')
        base_dir (str): Base directory containing the log files
    """
    print(f"\n{'='*60}")
    print(f"Processing {m_value}")
    print(f"{'='*60}")
    
    # Define file paths
    filtered_file = os.path.join(base_dir, 'filtered_top_candidate', f'Deep_KNNFirst_{m_value}.log')
    unfiltered_file = os.path.join(base_dir, 'unfiltered_top_candidate', f'Deep_KNNFirst_{m_value}.log')
    
    print(f"Analyzing filtered data from: {filtered_file}")
    print(f"Analyzing unfiltered data from: {unfiltered_file}")
    
    # Check if files exist
    if not os.path.exists(filtered_file):
        print(f"Error: Filtered file not found at {filtered_file}")
        return False
    
    if not os.path.exists(unfiltered_file):
        print(f"Error: Unfiltered file not found at {unfiltered_file}")
        return False
    
    # Parse the log files
    print(f"\nParsing log files for {m_value}...")
    filtered_data = parse_log_file(filtered_file)
    unfiltered_data = parse_log_file(unfiltered_file)
    
    print(f"Parsed {len(filtered_data)} ef values from filtered data")
    print(f"Parsed {len(unfiltered_data)} ef values from unfiltered data")
    
    # Create visualizations
    print(f"\nGenerating recall-QPS trade-off visualizations for {m_value}...")
    create_recall_qps_tradeoff_plots(filtered_data, unfiltered_data, m_value, base_dir)
    create_summary_comparison(filtered_data, unfiltered_data, m_value, base_dir)
    
    # Generate summary statistics
    print(f"\nGenerating summary statistics for {m_value}...")
    generate_summary_statistics(filtered_data, unfiltered_data, m_value, base_dir)
    
    print(f"\n{m_value} processing complete!")
    return True

def main():
    """
    Main function to orchestrate the visualization process for all M values.
    """
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Deep KNN-First Recall-QPS Trade-off Analysis")
    print("Multiple M Values Comparison (M16, M32, M64)")
    print("=" * 60)
    
    # M values to process
    m_values = ['M16', 'M32', 'M64']
    
    successful_m_values = []
    
    # Process each M value
    for m_value in m_values:
        success = process_m_value(m_value, base_dir)
        if success:
            successful_m_values.append(m_value)
    
    # Final summary
    print(f"\n{'='*60}")
    print("OVERALL PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {', '.join(successful_m_values)}")
    print(f"Files saved in: {base_dir}")
    print("\nGenerated files for each M value:")
    for m_value in successful_m_values:
        print(f"  {m_value}:")
        print(f"    - recall_qps_tradeoff_by_range_{m_value}.png")
        print(f"    - summary_comparison_{m_value}.png")
        print(f"    - tradeoff_analysis_summary_{m_value}.txt")

if __name__ == "__main__":
    main() 