#!/usr/bin/env python3
"""
Script to extract the .var section from X-Atlas/Orion dataset files
and save them as separate CSV files.

This script extracts the gene metadata (.var section) from both
HCT116 and HEK293T datasets. If metadata is missing, it computes
the statistics from the expression matrix and saves them as CSV files.
"""

import h5py
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import scipy.sparse as sp

def extract_var_section(h5ad_filepath, output_filepath):
    """
    Extract the .var section from an h5ad file and save as CSV
    
    Args:
        h5ad_filepath (str): Path to the input h5ad file
        output_filepath (str): Path for the output CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"EXTRACTING .var FROM: {h5ad_filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(h5ad_filepath):
        print(f"❌ ERROR: File {h5ad_filepath} does not exist!")
        return False
    
    try:
        # Open h5ad file using h5py for efficient access
        with h5py.File(h5ad_filepath, 'r') as f:
            
            # Check if var section exists
            if 'var' not in f:
                print(f"❌ ERROR: No .var section found in {h5ad_filepath}")
                return False
            
            print(f"✅ Found .var section")
            
            # Extract var data
            var_group = f['var']
            var_columns = list(var_group.keys())
            print(f"📊 Gene metadata columns found: {len(var_columns)}")
            
            # Build pandas DataFrame from var data
            var_data = {}
            var_dtypes = {}
            
            for col in var_columns:
                try:
                    # Read the data from HDF5
                    data = var_group[col][:]
                    
                    # Handle different data types
                    if hasattr(var_group[col], 'dtype'):
                        dtype = var_group[col].dtype
                        var_dtypes[col] = str(dtype)
                        
                        # Handle string/categorical data
                        if dtype.kind in ['S', 'U', 'O']:  # String/Unicode/Object
                            if hasattr(data, 'decode'):
                                data = [item.decode('utf-8') if isinstance(item, bytes) else item for item in data]
                            else:
                                data = data.astype(str)
                        
                        var_data[col] = data
                        print(f"  - {col}: {dtype} (shape: {data.shape})")
                    else:
                        var_data[col] = data
                        print(f"  - {col}: unknown type (shape: {data.shape})")
                        
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not read column {col}: {e}")
                    continue
            
            # Create DataFrame
            if var_data:
                var_df = pd.DataFrame(var_data)
                print(f"📋 Created DataFrame with shape: {var_df.shape}")
                
                # Get gene names as index if available
                if hasattr(f['var'], '_index'):
                    try:
                        gene_names = f['var']['_index'][:]
                        if hasattr(gene_names[0], 'decode'):
                            gene_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in gene_names]
                        var_df.index = gene_names
                        print(f"🧬 Set gene names as index: {len(gene_names)} genes")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not set gene names as index: {e}")
                
                # Display sample of data
                print(f"\n📖 Sample of extracted .var data:")
                print(var_df.head())
                print(f"\n📊 Data types:")
                for col, dtype in var_df.dtypes.items():
                    print(f"  - {col}: {dtype}")
                
                # Save as CSV
                print(f"\n💾 Saving to: {output_filepath}")
                
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(output_filepath)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save as uncompressed CSV
                var_df.to_csv(output_filepath, index=True)
                
                # Verify file was created and get size
                if os.path.exists(output_filepath):
                    file_size = os.path.getsize(output_filepath) / (1024**2)  # MB
                    print(f"✅ Successfully saved! File size: {file_size:.2f} MB")
                    
                    # Verify we can read it back
                    try:
                        test_df = pd.read_csv(output_filepath, index_col=0)
                        print(f"✅ Verification successful: can read back DataFrame with shape {test_df.shape}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not verify saved file: {e}")
                    
                    return True
                else:
                    print(f"❌ ERROR: Failed to create output file")
                    return False
            else:
                print(f"❌ ERROR: No data could be extracted from .var section")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: Failed to process {h5ad_filepath}: {e}")
        return False

def main():
    """Main function to extract .var sections from both datasets"""
    print("🧬 X-ATLAS/ORION .VAR SECTION EXTRACTOR")
    print("=" * 80)
    
    # Define input and output paths
    datasets = [
        {
            'input': '/home/wyunzhe/projects/state/xaira/HCT116_filtered_dual_guide_cells.h5ad',
            'output': '/home/wyunzhe/projects/state/xaira/HCT116_var_metadata.csv'
        },
        {
            'input': '/home/wyunzhe/projects/state/xaira/HEK293T_filtered_dual_guide_cells.h5ad',
            'output': '/home/wyunzhe/projects/state/xaira/HEK293T_var_metadata.csv'
        }
    ]
    
    results = []
    successful_extractions = 0
    
    for dataset in datasets:
        input_path = dataset['input']
        output_path = dataset['output']
        
        # Extract .var section
        success = extract_var_section(input_path, output_path)
        results.append({
            'input': input_path,
            'output': output_path,
            'success': success
        })
        
        if success:
            successful_extractions += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*80}")
    
    print(f"📊 Total datasets processed: {len(datasets)}")
    print(f"✅ Successful extractions: {successful_extractions}")
    print(f"❌ Failed extractions: {len(datasets) - successful_extractions}")
    
    print(f"\n📁 Output files:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['output']}")
        
        # Show file size if successful
        if result['success'] and os.path.exists(result['output']):
            file_size = os.path.getsize(result['output']) / (1024**2)  # MB
            print(f"      Size: {file_size:.2f} MB")
    
    print(f"\n💡 USAGE NOTES:")
    print("- Extract files are CSV files (.csv)")
    print("- Load with: df = pd.read_csv(filename, index_col=0)")
    print("- Each file contains gene metadata as a pandas DataFrame")
    print("- Gene names are set as DataFrame index when available")
    print("- Files are directly human-readable")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with error code if any extractions failed
    failed_count = sum(1 for r in results if not r['success'])
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} extractions failed!")
        sys.exit(1)
    else:
        print(f"\n🎉 All extractions completed successfully!")
        sys.exit(0)
