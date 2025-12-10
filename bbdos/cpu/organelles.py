"""
Neural ALU Organelles

Specialized micro-networks for arithmetic sub-tasks.
Each organelle has ONE job. No interference.

Architecture: Hierarchical Mixture of Experts (H-MoE)
Encoding: Soroban Spatial (32-bit Thermometer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abacus import SorobanEncoder


class Organelle_Result(nn.Module):
    """
    The Muscle - Computes 8-bit addition result.
    
    Input: A(32) + M(32) + C_in(1) = 65 dims
    Output: Result as 32-bit Soroban vector
    
    Learns: Bead shifts and ripple propagation
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(65, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 65] - Global input vector (A_sor, M_sor, C_in)
        Returns:
            [batch, 32] - Result logits (Soroban encoded)
        """
        return self.net(x)
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Organelle_C(nn.Module):
    """
    The Carry - Detects unsigned overflow.
    
    Input: A(32) + M(32) + C_in(1) = 65 dims
    Output: 1 bit (carry out logit)
    
    Learns: Threshold where A + M + C > 255
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(65, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 65] - Global input vector
        Returns:
            [batch, 1] - Carry out logit
        """
        return self.net(x)
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Organelle_V(nn.Module):
    """
    The Overflow - Detects signed overflow (two's complement).
    
    Input: A(32) + M(32) + C_in(1) = 65 dims
    Output: 1 bit (overflow logit)
    
    Learns: V = ~(A ^ M) & (A ^ R) & 0x80
    (Sign of A and M same, but result sign differs)
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(65, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 65] - Global input vector
        Returns:
            [batch, 1] - Overflow logit
        """
        return self.net(x)
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Organelle_NZ(nn.Module):
    """
    The Observers - Detects Zero and Negative flags.
    
    Input: A(32) + M(32) + C_in(1) = 65 dims
    Output: 2 bits (Z, N logits)
    
    Learns: Simple pattern matching on result
    - Z: Result == 0
    - N: Result & 0x80 != 0
    """
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(65, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [Z, N]
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 65] - Global input vector
        Returns:
            [batch, 2] - [Z_logit, N_logit]
        """
        return self.net(x)
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class OrganelleCluster(nn.Module):
    """
    The complete set of organelles for ADC.
    
    Bundles all organelles together for easy forward pass.
    Does NOT include the orchestrator - that's trained separately.
    """
    
    def __init__(self):
        super().__init__()
        
        self.soroban = SorobanEncoder()
        
        self.org_result = Organelle_Result(hidden_dim=256)
        self.org_c = Organelle_C(hidden_dim=64)
        self.org_v = Organelle_V(hidden_dim=64)
        self.org_nz = Organelle_NZ(hidden_dim=32)
    
    def encode_input(self, a, m, c_in):
        """
        Encode inputs to global input vector.
        
        Args:
            a: [batch] accumulator values (0-255)
            m: [batch] operand values (0-255)
            c_in: [batch] carry in (0 or 1)
        
        Returns:
            [batch, 65] global input vector
        """
        a_sor = self.soroban.encode_batch(a)  # [batch, 32]
        m_sor = self.soroban.encode_batch(m)  # [batch, 32]
        c = c_in.float().unsqueeze(-1)        # [batch, 1]
        
        return torch.cat([a_sor, m_sor, c], dim=-1)  # [batch, 65]
    
    def forward(self, a, m, c_in):
        """
        Forward pass through all organelles.
        
        Args:
            a: [batch] accumulator values (0-255)
            m: [batch] operand values (0-255)
            c_in: [batch] carry in (0 or 1)
        
        Returns:
            dict with raw logits from each organelle:
            - 'result': [batch, 32] Soroban logits
            - 'c': [batch, 1] carry logit
            - 'v': [batch, 1] overflow logit
            - 'nz': [batch, 2] [Z, N] logits
        """
        x = self.encode_input(a, m, c_in)
        
        return {
            'result': self.org_result(x),
            'c': self.org_c(x),
            'v': self.org_v(x),
            'nz': self.org_nz(x),
        }
    
    def forward_from_encoded(self, x):
        """Forward from pre-encoded input vector."""
        return {
            'result': self.org_result(x),
            'c': self.org_c(x),
            'v': self.org_v(x),
            'nz': self.org_nz(x),
        }
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def parameter_breakdown(self):
        """Return parameter count per organelle."""
        return {
            'org_result': self.org_result.num_parameters,
            'org_c': self.org_c.num_parameters,
            'org_v': self.org_v.num_parameters,
            'org_nz': self.org_nz.num_parameters,
            'total': self.num_parameters,
        }
