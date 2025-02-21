import torch
import torch.nn as nn
from frame_processor import CNNFeatureExtractor

class AdaptiveBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        # Skip normalization if batch size is 1.
        if input.shape[0] == 1:
            return input
        return super().forward(input)

class RLAgent(nn.Module):
    """
    A streamlined RL agent that uses CNN features combined with a lightweight fully-connected network.
    The LSTM has been removed to speed up backpropagation.
    """
    def __init__(self,
                 frame_stack: int = 10,
                 target_size: tuple = (128, 128),
                 num_actions: int = 10,
                 feature_dim: int = 256,
                 hidden_size: int = 512):
        """
        Args:
            frame_stack (int): Number of frames stacked together.
            target_size (tuple): Spatial dimensions for CNN input.
            num_actions (int): Size of the action space.
            feature_dim (int): Dimensionality of CNN features.
            hidden_size (int): Number of hidden units in the shared FC layers.
        """
        super(RLAgent, self).__init__()
        
        # CNN feature extractor remains the same.
        self.cnn = CNNFeatureExtractor(
            input_channels=frame_stack * 4,  # e.g., if you use 4 channels per frame.
            output_size=feature_dim,
            target_size=target_size
        )
        
        # Combine CNN features with extra features in a shared fully-connected layer.
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden_size),
            AdaptiveBatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Policy head: predicts logits for the action probabilities.
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            AdaptiveBatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Value head: predicts the state value.
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            AdaptiveBatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, state_tensor: torch.Tensor, extra_features: torch.Tensor):
        """
        Forward pass through the network.
        
        Args:
            state_tensor (torch.Tensor): Tensor of shape (batch, channels, H, W) from stacked frames.
            extra_features (torch.Tensor): Tensor of shape (batch, 4) with additional state information.
        
        Returns:
            policy_logits (torch.Tensor): Tensor of shape (batch, num_actions) with action scores.
            value (torch.Tensor): Tensor of shape (batch, 1) with state value predictions.
        """
        # Extract visual features using the CNN.
        cnn_features = self.cnn(state_tensor)  # (batch, feature_dim)
        
        # Concatenate CNN features with extra features.
        combined_features = torch.cat([cnn_features, extra_features], dim=1)  # (batch, feature_dim + 4)
        shared = self.shared_fc(combined_features)  # (batch, hidden_size)
        
        # Compute policy and value outputs.
        policy_logits = self.policy_head(shared)  # (batch, num_actions)
        value = self.value_head(shared)           # (batch, 1)
        
        return policy_logits, value

# Example usage:
if __name__ == "__main__":
    # Create dummy inputs.
    batch_size = 32
    channels = 10 * 4  # frame_stack * 4 channels
    height, width = 128, 128
    state_tensor = torch.randn(batch_size, channels, height, width)
    extra_features = torch.randn(batch_size, 4)
    
    # Instantiate and run the agent.
    agent = RLAgent(frame_stack=10, target_size=(height, width), num_actions=10)
    logits, value = agent(state_tensor, extra_features)
    print("Policy logits shape:", logits.shape)  # Expected: (32, 10)
    print("Value shape:", value.shape)            # Expected: (32, 1)
