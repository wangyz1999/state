#!/usr/bin/env python3
"""
Temporary script to inspect the contents of pickle and torch files
"""

import pickle
import torch
import os
from pathlib import Path

def inspect_pickle_file(filepath):
    """Inspect a pickle file and print its contents"""
    print(f"\n{'='*60}")
    print(f"INSPECTING PICKLE FILE: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ File does not exist: {filepath}")
        return
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Successfully loaded pickle file")
        print(f"📊 Data type: {type(data)}")
        print(f"📏 File size: {os.path.getsize(filepath)} bytes")
        
        if isinstance(data, dict):
            print(f"🔑 Dictionary with {len(data)} keys")
            print("Keys:")
            for i, key in enumerate(data.keys()):
                if i < 10:  # Show first 10 keys
                    print(f"  - {key}: {type(data[key])}")
                    if hasattr(data[key], 'shape'):
                        print(f"    Shape: {data[key].shape}")
                    elif isinstance(data[key], (list, tuple)):
                        print(f"    Length: {len(data[key])}")
                elif i == 10:
                    print(f"  ... and {len(data) - 10} more keys")
                    break
        elif isinstance(data, (list, tuple)):
            print(f"📋 {type(data).__name__} with {len(data)} items")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"First item shape: {data[0].shape}")
        elif hasattr(data, 'shape'):
            print(f"🔢 Array/Tensor shape: {data.shape}")
            print(f"Data type: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
        else:
            print(f"📄 Content preview: {str(data)[:200]}...")
            
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")

def inspect_torch_file(filepath):
    """Inspect a PyTorch file and print its contents"""
    print(f"\n{'='*60}")
    print(f"INSPECTING TORCH FILE: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ File does not exist: {filepath}")
        return
    
    try:
        # Try with weights_only=False first for older torch files
        try:
            data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception:
            # Fallback to default behavior
            data = torch.load(filepath, map_location='cpu')
        
        print(f"✅ Successfully loaded torch file")
        print(f"📊 Data type: {type(data)}")
        print(f"📏 File size: {os.path.getsize(filepath)} bytes")
        
        if isinstance(data, dict):
            print(f"🔑 Dictionary with {len(data)} keys")
            print("Keys:")
            for i, key in enumerate(data.keys()):
                if i < 10:  # Show first 10 keys
                    print(f"  - {key}: {type(data[key])}")
                    if hasattr(data[key], 'shape'):
                        print(f"    Shape: {data[key].shape}")
                    elif isinstance(data[key], (list, tuple)):
                        print(f"    Length: {len(data[key])}")
                elif i == 10:
                    print(f"  ... and {len(data) - 10} more keys")
                    break
        elif isinstance(data, torch.Tensor):
            print(f"🔢 Tensor shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Device: {data.device}")
            if data.numel() < 100:  # Show values for small tensors
                print(f"Values: {data}")
        elif isinstance(data, (list, tuple)):
            print(f"📋 {type(data).__name__} with {len(data)} items")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"First item shape: {data[0].shape}")
        else:
            print(f"📄 Content preview: {str(data)[:200]}...")
            
    except Exception as e:
        print(f"❌ Error loading torch file: {e}")

def inspect_checkpoint(filepath):
    """Inspect a PyTorch checkpoint and extract model architecture"""
    print(f"\n{'='*60}")
    print(f"INSPECTING CHECKPOINT: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ File does not exist: {filepath}")
        return
    
    try:
        # Try with weights_only=False first for older torch files
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception:
            # Fallback to default behavior
            checkpoint = torch.load(filepath, map_location='cpu')
        
        print(f"✅ Successfully loaded checkpoint")
        print(f"📊 Data type: {type(checkpoint)}")
        print(f"📏 File size: {os.path.getsize(filepath)} bytes")
        
        if isinstance(checkpoint, dict):
            print(f"\n🔑 Checkpoint contains {len(checkpoint)} top-level keys:")
            for key in checkpoint.keys():
                print(f"  - {key}")
            
            # Look for common checkpoint keys
            if 'state_dict' in checkpoint:
                print(f"\n🏗️  MODEL STATE_DICT ANALYSIS:")
                state_dict = checkpoint['state_dict']
                print(f"Number of parameters: {len(state_dict)}")
                
                # Group parameters by layer/module
                layer_info = {}
                total_params = 0
                
                for param_name, param_tensor in state_dict.items():
                    # Extract layer name (everything before the last dot)
                    if '.' in param_name:
                        layer_name = '.'.join(param_name.split('.')[:-1])
                    else:
                        layer_name = param_name
                    
                    if layer_name not in layer_info:
                        layer_info[layer_name] = {'params': [], 'total_params': 0}
                    
                    layer_info[layer_name]['params'].append({
                        'name': param_name,
                        'shape': param_tensor.shape,
                        'numel': param_tensor.numel()
                    })
                    layer_info[layer_name]['total_params'] += param_tensor.numel()
                    total_params += param_tensor.numel()
                
                print(f"Total parameters: {total_params:,}")
                print(f"\n📋 Layer breakdown (showing first 20 layers):")
                
                for i, (layer_name, info) in enumerate(layer_info.items()):
                    if i >= 20:
                        print(f"... and {len(layer_info) - 20} more layers")
                        break
                    print(f"  {layer_name}: {info['total_params']:,} params")
                    for param in info['params'][:3]:  # Show first 3 params per layer
                        print(f"    - {param['name']}: {param['shape']}")
                    if len(info['params']) > 3:
                        print(f"    ... and {len(info['params']) - 3} more parameters")
            
            # Look for model architecture info
            if 'model' in checkpoint:
                print(f"\n🏛️  MODEL OBJECT:")
                model = checkpoint['model']
                print(f"Model type: {type(model)}")
                if hasattr(model, '__dict__'):
                    print("Model attributes:")
                    for attr, value in model.__dict__.items():
                        if not attr.startswith('_'):
                            print(f"  - {attr}: {type(value)}")
            
            # Look for hyperparameters
            if 'hyper_parameters' in checkpoint:
                print(f"\n⚙️  HYPERPARAMETERS:")
                hparams = checkpoint['hyper_parameters']
                for key, value in hparams.items():
                    print(f"  - {key}: {value}")
            
            # Look for other useful info
            for key in ['epoch', 'global_step', 'pytorch-lightning_version', 'lr_schedulers', 'optimizer_states']:
                if key in checkpoint:
                    print(f"\n📊 {key.upper()}:")
                    value = checkpoint[key]
                    if isinstance(value, (int, float, str)):
                        print(f"  {value}")
                    elif isinstance(value, dict) and len(value) < 10:
                        for k, v in value.items():
                            print(f"  - {k}: {v}")
                    else:
                        print(f"  Type: {type(value)}, Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def main():
    files_to_inspect = [
        "/home/wyunzhe/projects/state/competition/state_sm_tahoe_from_scratch/batch_onehot_map.pkl",
        "/home/wyunzhe/projects/state/competition/state_sm_tahoe_from_scratch/pert_onehot_map.pt",
        "/home/wyunzhe/projects/state/competition/state_sm_tahoe_from_scratch/cell_type_onehot_map.pkl",
        "/home/wyunzhe/projects/state/tahoe_final.ckpt"
    ]
    
    print("🔍 INSPECTING FILES")
    print("=" * 80)
    
    for filepath in files_to_inspect:
        if filepath.endswith('.pkl'):
            inspect_pickle_file(filepath)
        elif filepath.endswith('.pt'):
            inspect_torch_file(filepath)
        elif filepath.endswith('.ckpt'):
            inspect_checkpoint(filepath)
        else:
            print(f"\n⚠️  Unknown file format: {filepath}")
    
    print(f"\n{'='*80}")
    print("✅ INSPECTION COMPLETE")

if __name__ == "__main__":
    main()