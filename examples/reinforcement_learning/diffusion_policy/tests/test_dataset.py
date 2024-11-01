import os
import sys
import torch
import traceback
from config import DataConfig, ModelConfig
from dataset import SequentialDataset
from model import ObservationEncoder, create_model



# add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def test_config():
    """Test configuration creation"""
    print("\nTesting Config...")

    data_config = DataConfig(
        dataset_path=os.path.join(parent_dir, "pusht_cchi_v7_replay.zarr"),
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        state_dim=5,
        action_dim=2
    )

    model_config = ModelConfig()

    print("Data Config:")
    print(f"- pred_horizon: {data_config.pred_horizon}")
    print(f"- obs_horizon: {data_config.obs_horizon}")
    print(f"- action_dim: {data_config.action_dim}")
    print(f"- state_dim: {data_config.state_dim}")

    print("\nModel Config:")
    print(f"- obs_embed_dim: {model_config.obs_embed_dim}")
    print(f"- total_in_channels: {model_config.total_in_channels}")

    return data_config, model_config

def test_dataset(data_config):
    """Test dataset loading and processing"""
    print("\nTesting Dataset...")

    try:
        dataset = SequentialDataset(data_config)
        print(f"Dataset size: {len(dataset)}")

        # get one sample
        sample = dataset[0]
        print("\nSample shapes:")
        for key, value in sample.items():
            print(f"- {key}: {value.shape}")
            print(f"- {key} range: [{value.min():.3f}, {value.max():.3f}]")

        # test dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, num_workers=2,
            shuffle=True
        )

        batch = next(iter(dataloader))
        print("\nBatch shapes:")
        for key, value in batch.items():
            print(f"- {key}: {value.shape}")

        return dataset, dataloader

    except Exception as e:
        print(f"Error in dataset testing: {e}")
        traceback.print_exc()
        return None, None

def test_observation_encoder(data_config, model_config):
    """Test observation encoder"""
    print("\nTesting Observation Encoder...")

    try:
        # encoder
        encoder = ObservationEncoder(
            obs_dim=data_config.state_dim,
            embed_dim=model_config.obs_embed_dim
        )

        # dummy batch of observations
        batch_size = 32
        obs = torch.randn(batch_size, data_config.obs_horizon, data_config.state_dim)

        # observations
        encoded = encoder(obs)
        print(f"Input shape: {obs.shape}")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Expected shape: [batch={batch_size}, {data_config.obs_horizon * model_config.obs_embed_dim}]")

        return encoder

    except Exception as e:
        print(f"Error in encoder testing: {e}")
        traceback.print_exc()
        return None

def test_model_creation(data_config, model_config):
    """Test full model creation"""
    print("\nTesting Model Creation...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # models
        model, obs_encoder, obs_projection, noise_scheduler = create_model(
            data_config, model_config, device
        )

        print("\nModel components created:")
        print(f"- UNet1D parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"- Encoder parameters: {sum(p.numel() for p in obs_encoder.parameters())}")
        print(f"- Projection parameters: {sum(p.numel() for p in obs_projection.parameters())}")

        # forward pass
        batch_size = 32
        obs = torch.randn(batch_size, data_config.obs_horizon, data_config.state_dim).to(device)
        actions = torch.randn(batch_size, data_config.pred_horizon, data_config.action_dim).to(device)

        print("\nTesting forward pass...")

        # 1. Encode observations
        obs_embedding = obs_encoder(obs)
        print(f"Obs embedding shape: {obs_embedding.shape}")

        # 2. Project observations
        obs_cond = obs_projection(obs_embedding)
        print(f"Obs projection shape: {obs_cond.shape}")

        # 3. Add noise to actions
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

        # 4. Prepare for UNet (reshape)
        noisy_actions = noisy_actions.transpose(1, 2)
        obs_cond = obs_cond.unsqueeze(-1).expand(-1, -1, noisy_actions.shape[-1])

        # 5. Concatenate and pass through model
        model_input = torch.cat([noisy_actions, obs_cond], dim=1)
        print(f"Model input shape: {model_input.shape}")

        output = model(model_input, timesteps).sample
        print(f"Model output shape: {output.shape}")

        print("\nAll forward passes successful!")

        return model, obs_encoder, obs_projection, noise_scheduler

    except Exception as e:
        print(f"Error in model testing: {e}")
        traceback.print_exc()
        return None, None, None, None

def main():
    # run all tests
    print("Starting tests...")

    data_config, model_config = test_config()
    dataset, dataloader = test_dataset(data_config)

    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
