import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOClassifier(nn.Module):
    
    def __init__(self, 
                 model_type="s",
                 hidden_layer_dims=[],
                 use_dropout=False,
                 dropout_prob=0.5,
                 device="cuda"):
        super(DINOClassifier, self).__init__()

        backbone_types = ["s", "b"]
        backbone_out_size = { "s": 384, "b": 768 }
        assert model_type in backbone_types, "Model architecture not available."

        self.model_type = model_type
        self.model_name = f"dino_vit{self.model_type}16"


        self.backbone = torch.hub.load('facebookresearch/dino:main', self.model_name,
                                       force_reload=False).to(device) 

        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.dropout_prob = dropout_prob if use_dropout else 0.0

        fc_input_dim = backbone_out_size[model_type]
        self.fc_head = nn.ModuleList()

        for dim in hidden_layer_dims:
            self.fc_head.append(nn.Linear(fc_input_dim, dim))
            fc_input_dim = dim  # Updateing input dimension for the next layer
            self.fc_head.append(nn.Dropout(p=dropout_prob))

        self.fc_head.append(nn.Linear(fc_input_dim, 1))

        self.fc_head = nn.Sequential(*self.fc_head)

        self.fc_head.to(device)

    def forward(self, x):
        # Getting features from ViT-Backbone
        features = self.backbone(x)

        # Processing the features in the fully connected classification head
        feature_input = features.view(features.size(0), -1)

        for layer in self.fc_head[:-1]:  # Exclude the last layer
            feature_input = layer(feature_input)
            if isinstance(layer, nn.Linear):
                feature_input = F.relu(feature_input)

        # Apply sigmoid activation directly on the final layer's output
        out = torch.sigmoid(self.fc_head[-1](feature_input))

        return out, features