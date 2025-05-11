# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLSTM(nn.Module):
    def __init__(self,
                 resnet_model_name: str = 'resnet18',
                 pretrained: bool = True,
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 output_dim: int = 1): # Output dim 1 for regression (intensity value)
        """
        Combines a ResNet feature extractor with an LSTM for sequence processing.

        Args:
            resnet_model_name (str): Name of the ResNet model to use (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained weights for ResNet.
            lstm_hidden_size (int): Number of features in the LSTM hidden state.
            lstm_num_layers (int): Number of recurrent layers in the LSTM.
            lstm_dropout (float): Dropout probability for LSTM layers (if num_layers > 1).
            output_dim (int): Dimension of the final output (e.g., 1 for intensity regression).
        """
        super().__init__()

        # --- ResNet Feature Extractor ---
        if resnet_model_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        elif resnet_model_name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
        elif resnet_model_name == 'resnet50':
             weights = models.ResNet50_Weights.DEFAULT if pretrained else None
             resnet = models.resnet50(weights=weights)
        # Add more options (resnet101, etc.) if needed
        else:
            raise ValueError(f"Unsupported ResNet model: {resnet_model_name}")

        # Modify the first convolutional layer to accept 1 input channel (grayscale)
        # instead of 3 (RGB). We can average the weights of the original 3 channels.
        original_conv1_weights = resnet.conv1.weight.data
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Average weights across the channel dimension
            new_conv1.weight.data = torch.mean(original_conv1_weights, dim=1, keepdim=True)
        resnet.conv1 = new_conv1

        # Remove the final fully connected layer (classification layer)
        modules = list(resnet.children())[:-1] # Remove the final fc layer
        self.resnet_features = nn.Sequential(*modules)

        # Determine the output feature size from ResNet
        # We run a dummy tensor through the feature extractor
        dummy_input = torch.randn(1, 1, 224, 224) # ResNet typically expects 224x224, but works with others
                                                 # The adaptive avg pool handles size variations
        resnet_output_size = self.resnet_features(dummy_input).view(1, -1).size(1)


        # --- LSTM Layer ---
        self.lstm = nn.LSTM(input_size=resnet_output_size,
                              hidden_size=lstm_hidden_size,
                              num_layers=lstm_num_layers,
                              batch_first=True, # Input shape: (batch, seq, feature)
                              dropout=lstm_dropout if lstm_num_layers > 1 else 0,
                              bidirectional=False) # Standard forward LSTM

        # --- Output Layer ---
        # Maps LSTM hidden state to the final prediction dimension
        self.fc_out = nn.Linear(lstm_hidden_size, output_dim)        # Maps LSTM hidden state to the final prediction dimension
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5),                      # Dropout before final linear layer
            nn.Linear(lstm_hidden_size, output_dim)  # Final output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet-LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        batch_size, seq_len, C, H, W = x.size()

        # Reshape input for ResNet: (batch_size * sequence_length, C, H, W)
        x_reshaped = x.view(batch_size * seq_len, C, H, W)

        # Extract features using ResNet
        # Output shape: (batch_size * sequence_length, resnet_output_size, 1, 1) after avgpool
        features = self.resnet_features(x_reshaped)
        # Flatten features: (batch_size * sequence_length, resnet_output_size)
        features = features.view(batch_size * seq_len, -1)

        # Reshape features back into sequence format for LSTM: (batch_size, sequence_length, resnet_output_size)
        features_seq = features.view(batch_size, seq_len, -1)

        # Pass features through LSTM
        # Output shape: (batch_size, sequence_length, lstm_hidden_size)
        lstm_out, _ = self.lstm(features_seq) # We only need the output sequence, not the final hidden state

        # Pass LSTM output through the final fully connected layer
        # Input shape: (batch_size * sequence_length, lstm_hidden_size) - apply Linear layer to each time step
        # We can reshape or apply the linear layer directly if batch_first=True
        output = self.fc_out(lstm_out) # Output shape: (batch_size, sequence_length, output_dim)

        return output