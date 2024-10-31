import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import DataConfig, ModelConfig
from train import train_diffusion

def test_training_setup():
    """Test training initialization"""
    print("\nTesting training setup...")
    
    # Create configs
    data_config = DataConfig(
        dataset_path=os.path.join(parent_dir, "pusht_cchi_v7_replay.zarr"),
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        state_dim=5,
        action_dim=2
    )
    
    model_config = ModelConfig()
    
    print("Configurations created successfully")
    return data_config, model_config

def test_mini_training(data_config, model_config):
    """Test a few iterations of training"""
    print("\nTesting mini training...")
    
    try:
        # Run training for just 2 epochs with small batch
        results = train_diffusion(
            data_config=data_config,
            model_config=model_config,
            num_epochs=2,
            batch_size=32,
            save_dir=os.path.join(current_dir, "test_checkpoints")
        )
        
        print("Mini training completed successfully")
        
        # Check if all components are returned
        required_keys = ['model', 'obs_encoder', 'obs_projection', 
                        'ema', 'noise_scheduler', 'optimizer', 'stats']
        
        for key in required_keys:
            assert key in results, f"Missing {key} in training results"
        
        print("All model components returned correctly")
        
        # Check if checkpoints were saved
        checkpoint_path = os.path.join(current_dir, "test_checkpoints", "diffusion_final.pt")
        assert os.path.exists(checkpoint_path), "Final checkpoint not saved"
        
        print("Checkpoints saved successfully")
        
        return results
    
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_inference(results):
    """Test if trained model can do inference"""
    print("\nTesting model inference...")
    
    try:
        # Get components
        model = results['model']
        obs_encoder = results['obs_encoder']
        obs_projection = results['obs_projection']
        
        # Create dummy data
        batch_size = 1
        state = torch.randn(batch_size, 2, 5)  # [batch, obs_horizon, state_dim]
        
        # Run inference
        with torch.no_grad():
            # 1. Encode observation
            obs_embedding = obs_encoder(state)
            print(f"Observation embedding shape: {obs_embedding.shape}")
            
            # 2. Project observation
            obs_cond = obs_projection(obs_embedding)
            print(f"Observation projection shape: {obs_cond.shape}")
            
            # 3. Prepare for model (just checking shapes)
            obs_cond = obs_cond.unsqueeze(-1).expand(-1, -1, 16)  # 16 = pred_horizon
            print(f"Expanded conditioning shape: {obs_cond.shape}")
            
            # 4. Create dummy noisy actions
            noisy_actions = torch.randn(batch_size, 2, 16)  # [batch, action_dim, pred_horizon]
            model_input = torch.cat([noisy_actions, obs_cond], dim=1)
            
            # 5. Run model
            timesteps = torch.zeros(batch_size, dtype=torch.long)
            output = model(model_input, timesteps).sample
            print(f"Model output shape: {output.shape}")
            
        print("Inference test successful")
        return True
        
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting training tests...")
    
    # Test setup
    data_config, model_config = test_training_setup()
    
    # Test training
    results = test_mini_training(data_config, model_config)
    
    if results is not None:
        # Test inference
        test_model_inference(results)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()