import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGrouping(nn.Module):
    """
    Learnable Variate Grouping Layer
    assigns N variates into G groups using a soft-assignment mechanism.
    """
    def __init__(self, n_vars, n_groups, d_model, dropout=0.1):
        super(LearnableGrouping, self).__init__()
        self.n_vars = n_vars
        self.n_groups = n_groups
        self.d_model = d_model
        
        # Group assignment scoring layer
        # Maps embedded variate (d_model) to group scores (n_groups)
        self.score_projection = nn.Linear(d_model, n_groups)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (B, N, E) where N is number of variates, E is d_model
        returns:
            group_representatives: (B, G, E)
            assignment_weights: (B, N, G)
        """
        B, N, E = x.shape
        G = self.n_groups
        
        # 1. Compute assignment scores with Logits Dropout (Expert Option 1)
        logits = self.score_projection(x) # (B, N, G)
        logits = self.dropout(logits)
        weights = F.softmax(logits, dim=-1) # (B, N, G)
        
        # 2. Compute group representatives with scaling normalization
        # group_representatives = weights.transpose(-1, -2) @ x
        group_representatives = torch.matmul(weights.transpose(-1, -2), x) # (B, G, E)
        
        # Expert recommendation: Scale by weight sum to stabilize training
        norm = weights.sum(dim=1, keepdim=True).transpose(-1, -2) + 1e-6
        group_representatives = group_representatives / norm
        
        return group_representatives, weights

class VariateReconstruction(nn.Module):
    """
    Broadcasts group-level features back to individual variates
    """
    def __init__(self, d_model):
        super(VariateReconstruction, self).__init__()
        # Optional: refinement layer after broadcasting
        self.refinement = nn.Linear(d_model, d_model)
        
    def forward(self, group_features, weights):
        """
        group_features: (B, G, E)
        weights: (B, N, G)
        returns: (B, N, E)
        """
        # Broadcast group features back using assignment weights
        # (B, N, G) @ (B, G, E) -> (B, N, E)
        x_reconstructed = torch.matmul(weights, group_features)
        return self.refinement(x_reconstructed)
