#!/usr/bin/env python3
"""
UMAP visualization of gene embeddings with meaningful biological labels per dataset
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import umap
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set up scanpy
sc.settings.verbosity = 1

def load_dataset(filepath, sample_size=None):
    """Load and optionally sample a single dataset"""
    print(f"  Loading {filepath.name}...")
    adata = sc.read_h5ad(filepath)
    
    if sample_size and sample_size < adata.n_obs:
        indices = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata_sampled = adata[indices].copy()
    else:
        adata_sampled = adata.copy()
    
    print(f"    Using {adata_sampled.n_obs} cells from {adata.n_obs}")
    return adata_sampled

def run_umap_for_dataset(X_state, n_neighbors=15, min_dist=0.1, random_state=42):
    """Run UMAP dimensionality reduction for a single dataset"""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=False  # Less verbose for individual datasets
    )
    
    embedding = reducer.fit_transform(X_state)
    return embedding

def get_meaningful_colorings(obs_df, dataset_name):
    """Define meaningful coloring criteria for each dataset"""
    
    # Common colorings for all datasets - put gene transcript at the end
    colorings = []
    
    # 1. Control vs Perturbation
    if 'target_gene' in obs_df.columns:
        obs_df['is_control'] = (obs_df['target_gene'] == 'non-targeting').astype(str)
        obs_df['is_control'] = obs_df['is_control'].map({'True': 'Control', 'False': 'Perturbation'})
        colorings.append(('is_control', 'Control vs Perturbation', 'categorical'))
    
    # 2. Batch effects (important for technical variation)
    if 'batch_var' in obs_df.columns:
        n_batches = obs_df['batch_var'].nunique()
        print(f"    Found {n_batches} unique batches")
        colorings.append(('batch_var', f'Batch Variable ({n_batches} batches)', 'categorical_top'))
    
    # 3. Top target genes (most relevant)
    if 'target_gene' in obs_df.columns:
        n_targets = obs_df['target_gene'].nunique()
        print(f"    Found {n_targets} unique target genes")
        colorings.append(('target_gene', f'Target Genes ({n_targets} total)', 'categorical_top'))
    
    # 4. UMI count (technical quality)
    if 'UMI_count' in obs_df.columns:
        colorings.append(('UMI_count', 'UMI Count', 'continuous'))
    
    # 5. Mitochondrial percentage (cell quality)
    if 'mitopercent' in obs_df.columns:
        colorings.append(('mitopercent', 'Mitochondrial %', 'continuous'))
    
    # 6. Gene transcript (rightmost - important for biological variation)
    if 'gene_transcript' in obs_df.columns:
        n_transcripts = obs_df['gene_transcript'].nunique()
        print(f"    Found {n_transcripts} unique gene transcripts")
        colorings.append(('gene_transcript', f'Gene Transcript ({n_transcripts} variants)', 'categorical_top_trimmed'))
    
    return colorings

def plot_dataset_umap(adata, dataset_name, ax_row, fig, sample_size=10000):
    """Create UMAP visualization for a single dataset"""
    
    # Sample data if needed
    if sample_size and sample_size < adata.n_obs:
        indices = np.random.choice(adata.n_obs, sample_size, replace=False)
        adata_subset = adata[indices].copy()
    else:
        adata_subset = adata.copy()
    
    print(f"  Running UMAP for {dataset_name} ({adata_subset.n_obs} cells)...")
    
    # Run UMAP
    embedding = run_umap_for_dataset(adata_subset.X)
    # embedding = run_umap_for_dataset(adata_subset.obsm['X_state'])
    
    # Get meaningful colorings
    colorings = get_meaningful_colorings(adata_subset.obs, dataset_name)
    
    # Limit to 6 most meaningful plots per dataset (to include batch and transcript)
    colorings = colorings[:6]
    
    for col_idx, (col, title, plot_type) in enumerate(colorings):
        ax = ax_row[col_idx]
        
        if col not in adata_subset.obs.columns:
            ax.text(0.5, 0.5, f'{col}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dataset_name}\n{title}', fontsize=10, fontweight='bold')
            continue
        
        if plot_type == 'categorical':
            # All categories
            unique_vals = adata_subset.obs[col].unique()
            n_unique = len(unique_vals)
            
            if n_unique <= 10:
                colors = plt.cm.Set3(np.linspace(0, 1, n_unique))
                for j, val in enumerate(unique_vals):
                    mask = adata_subset.obs[col] == val
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                             c=[colors[j]], label=val, alpha=0.7, s=0.8)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
            else:
                # Too many categories, use continuous coloring
                le = LabelEncoder()
                colors = le.fit_transform(adata_subset.obs[col])
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                                   c=colors, alpha=0.7, s=0.8, cmap='tab20')
        
        elif plot_type == 'categorical_top':
            # Show only top categories
            top_cats = adata_subset.obs[col].value_counts().head(10).index
            obs_subset = adata_subset.obs[col].isin(top_cats)
            
            unique_vals = adata_subset.obs.loc[obs_subset, col].unique()
            n_unique = len(unique_vals)
            
            # Plot others in gray first
            others_mask = ~obs_subset
            if others_mask.sum() > 0:
                ax.scatter(embedding[others_mask, 0], embedding[others_mask, 1], 
                         c='lightgray', alpha=0.3, s=0.5, label='Others')
            
            # Plot top categories
            colors = plt.cm.Set3(np.linspace(0, 1, n_unique))
            for j, val in enumerate(unique_vals):
                mask = adata_subset.obs[col] == val
                if mask.sum() > 0:
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                             c=[colors[j]], label=val, alpha=0.8, s=0.8)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        
        elif plot_type == 'categorical_top_trimmed':
            # Show only top categories with trimmed labels for gene transcripts
            top_cats = adata_subset.obs[col].value_counts().head(10).index
            obs_subset = adata_subset.obs[col].isin(top_cats)
            
            unique_vals = adata_subset.obs.loc[obs_subset, col].unique()
            n_unique = len(unique_vals)
            
            # Plot others in gray first
            others_mask = ~obs_subset
            if others_mask.sum() > 0:
                ax.scatter(embedding[others_mask, 0], embedding[others_mask, 1], 
                         c='lightgray', alpha=0.3, s=0.5, label='Others')
            
            # Plot top categories with trimmed labels
            colors = plt.cm.Set3(np.linspace(0, 1, n_unique))
            for j, val in enumerate(unique_vals):
                mask = adata_subset.obs[col] == val
                if mask.sum() > 0:
                    # Trim long labels to 10 characters + "..."
                    display_label = val if len(str(val)) <= 10 else str(val)[:10] + "..."
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                             c=[colors[j]], label=display_label, alpha=0.8, s=0.8)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        
        elif plot_type == 'continuous':
            if adata_subset.obs[col].dtype in ['object', 'category']:
                # Convert to numeric if possible
                try:
                    values = pd.to_numeric(adata_subset.obs[col])
                except:
                    ax.text(0.5, 0.5, f'{col}\ncannot convert to numeric', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{dataset_name}\n{title}', fontsize=10, fontweight='bold')
                    continue
            else:
                values = adata_subset.obs[col]
            
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=values, alpha=0.7, s=0.8, cmap='viridis')
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        ax.set_title(f'{dataset_name}\n{title}', fontsize=10, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=8)
        ax.set_ylabel('UMAP 2', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots in this row
    for col_idx in range(len(colorings), 6):
        ax_row[col_idx].set_visible(False)

def create_dataset_specific_visualization(data_dir, sample_size_per_dataset=10000, output_path='umap_X_count_per_dataset_analysis.pdf'):
    """Create UMAP visualizations for each dataset separately"""
    
    data_files = [
        'hepg2.h5ad',
        'jurkat.h5ad', 
        'k562.h5ad',
        'rpe1.h5ad',
        'k562_gwps.h5ad',
        'competition_train.h5ad'
    ]
    
    # Filter existing files
    existing_files = []
    for file in data_files:
        filepath = Path(data_dir) / file
        if filepath.exists():
            existing_files.append((filepath, file.replace('.h5ad', '')))
    
    print(f"Found {len(existing_files)} datasets")
    
    # Create figure with subplots: each row is a dataset, each column is a coloring
    n_datasets = len(existing_files)
    n_cols = 6  # Max 6 colorings per dataset (to include batch and transcript)
    
    fig, axes = plt.subplots(n_datasets, n_cols, figsize=(5*n_cols, 4*n_datasets))
    
    # Handle case where we have only one dataset
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.5)
    
    print("\nProcessing datasets...")
    for row_idx, (filepath, dataset_name) in enumerate(existing_files):      
        print(f"\nDataset {row_idx+1}/{n_datasets}: {dataset_name}")
        
        # Load dataset
        adata = load_dataset(filepath, sample_size_per_dataset)
        
        # Create UMAP visualization for this dataset
        ax_row = axes[row_idx] if n_datasets > 1 else axes[0]
        plot_dataset_umap(adata, dataset_name, ax_row, fig, sample_size_per_dataset)
        
        # Print dataset summary
        print(f"    Cell type: {adata.obs['cell_type'].unique()}")
        print(f"    Target genes: {adata.obs['target_gene'].nunique()} unique")
        print(f"    Controls: {(adata.obs['target_gene'] == 'non-targeting').sum()}")
        print(f"    Perturbed: {(adata.obs['target_gene'] != 'non-targeting').sum()}")
    
    plt.tight_layout(pad=3.0)
    
    try:
        if output_path.endswith('.pdf'):
            plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nVisualization saved to {output_path}")
        
        # Get file size for confirmation
        import os
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size/1024/1024:.2f} MB")
        
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close(fig)

def main():
    data_dir = "data_state_emb"
    sample_size_per_dataset = 3000  # Sample size per individual dataset
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Creating dataset-specific UMAP visualizations...")
    create_dataset_specific_visualization(data_dir, sample_size_per_dataset)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()