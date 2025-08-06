import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def read_metrics_progression():
    """Read metrics from all eval folders and return progression data."""
    base_path = Path("competition/state_sm_tahoe")
    
    # Generate step numbers from 2000 to 48000 in increments of 2000
    steps = list(range(2000, 50000, 2000))
    
    # Initialize lists to store metric values
    overlap_at_n_values = []
    mae_values = []
    discrimination_score_l1_values = []
    valid_steps = []
    
    for step in steps:
        eval_folder = base_path / f"eval_{step}"
        agg_results_file = eval_folder / "agg_results.csv"
        
        if agg_results_file.exists():
            try:
                # Read the CSV file
                df = pd.read_csv(agg_results_file)
                
                # Find the row with 'mean' statistic
                mean_row = df[df['statistic'] == 'mean']
                
                if not mean_row.empty:
                    overlap_at_n_values.append(mean_row['overlap_at_N'].iloc[0])
                    mae_values.append(mean_row['mae'].iloc[0])
                    discrimination_score_l1_values.append(mean_row['discrimination_score_l1'].iloc[0])
                    valid_steps.append(step)
                    print(f"Successfully read metrics for step {step}")
                else:
                    print(f"No 'mean' row found in {agg_results_file}")
            except Exception as e:
                print(f"Error reading {agg_results_file}: {e}")
        else:
            print(f"File not found: {agg_results_file}")
    
    return valid_steps, overlap_at_n_values, mae_values, discrimination_score_l1_values

def plot_metrics_progression():
    """Create and save the metrics progression plot."""
    # Read the data
    steps, overlap_values, mae_values, discrimination_values = read_metrics_progression()
    
    if not steps:
        print("No data found to plot!")
        return
    
    # Create the plot with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Metrics Progression Over Training Steps', fontsize=16, fontweight='bold')
    
    # Plot overlap_at_N
    axes[0].plot(steps, overlap_values, 'b-o', linewidth=2, markersize=6)
    axes[0].set_title('Overlap at N', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Overlap at N', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(min(steps)-1000, max(steps)+1000)
    
    # Plot MAE
    axes[1].plot(steps, mae_values, 'r-s', linewidth=2, markersize=6)
    axes[1].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(min(steps)-1000, max(steps)+1000)
    
    # Plot discrimination_score_l1
    axes[2].plot(steps, discrimination_values, 'g-^', linewidth=2, markersize=6)
    axes[2].set_title('Discrimination Score L1', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Discrimination Score L1', fontsize=10)
    axes[2].set_xlabel('Training Steps', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(min(steps)-1000, max(steps)+1000)
    
    # Format x-axis to show steps nicely
    for ax in axes:
        ax.set_xticks(range(2000, 50000, 4000))  # Show every 4000 steps
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('metrics_progression.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'metrics_progression.png'")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Steps analyzed: {len(steps)}")
    print(f"Step range: {min(steps)} - {max(steps)}")
    print(f"Overlap at N - Min: {min(overlap_values):.6f}, Max: {max(overlap_values):.6f}")
    print(f"MAE - Min: {min(mae_values):.6f}, Max: {max(mae_values):.6f}")
    print(f"Discrimination Score L1 - Min: {min(discrimination_values):.6f}, Max: {max(discrimination_values):.6f}")
    
    plt.close()

if __name__ == "__main__":
    plot_metrics_progression() 