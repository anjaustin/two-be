"""
Neural ALU Orchestrator (The Neural Bus)

Consistency enforcement and error correction layer.
Takes raw organelle outputs and cleans them up.

Enforced Logic:
- If Result == 0, boost Z to 1
- If massive ripple detected, boost C
- Resolve conflicting signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADC_Orchestrator(nn.Module):
    """
    The Neural Bus - Consistency Enforcer.
    
    Input: Concatenated logits from all organelles
           Result(32) + C(1) + V(1) + NZ(2) = 36 signals
    
    Output: Cleaned/corrected 36 signals
    
    Architecture: Densely Connected Residual Block
    Operation: Y_final = Bus(Y_raw) + Y_raw
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Input: 36 signals (32 result + 4 flags)
        self.bus = nn.Sequential(
            nn.Linear(36, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 36),
        )
    
    def forward(self, organelle_outputs):
        """
        Apply consistency corrections to organelle outputs.
        
        Args:
            organelle_outputs: dict with keys 'result', 'c', 'v', 'nz'
                - 'result': [batch, 32]
                - 'c': [batch, 1]
                - 'v': [batch, 1]
                - 'nz': [batch, 2]
        
        Returns:
            dict with corrected outputs (same structure)
        """
        # Concatenate all signals
        y_raw = torch.cat([
            organelle_outputs['result'],  # 32
            organelle_outputs['c'],       # 1
            organelle_outputs['v'],       # 1
            organelle_outputs['nz'],      # 2
        ], dim=-1)  # [batch, 36]
        
        # Apply bus (residual connection)
        correction = self.bus(y_raw)
        y_final = y_raw + correction
        
        # Split back to components
        return {
            'result': y_final[:, :32],
            'c': y_final[:, 32:33],
            'v': y_final[:, 33:34],
            'nz': y_final[:, 34:36],
        }
    
    def forward_flat(self, y_raw):
        """
        Forward from flat concatenated tensor.
        
        Args:
            y_raw: [batch, 36] concatenated organelle outputs
        
        Returns:
            [batch, 36] corrected outputs
        """
        correction = self.bus(y_raw)
        return y_raw + correction
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralALU(nn.Module):
    """
    Complete Neural ALU - Organelles + Orchestrator.
    
    This is the full ADC implementation with consistency enforcement.
    """
    
    def __init__(self):
        super().__init__()
        
        from .organelles import OrganelleCluster
        from .abacus import SorobanEncoder
        
        self.organelles = OrganelleCluster()
        self.orchestrator = ADC_Orchestrator(hidden_dim=64)
        self.soroban = SorobanEncoder()
    
    def forward(self, a, m, c_in, use_orchestrator=True):
        """
        Full forward pass: Organelles -> Orchestrator -> Output.
        
        Args:
            a: [batch] accumulator values (0-255)
            m: [batch] operand values (0-255)
            c_in: [batch] carry in (0 or 1)
            use_orchestrator: Whether to apply consistency correction
        
        Returns:
            dict with logits: 'result', 'c', 'v', 'nz'
        """
        # Get raw organelle outputs
        raw_outputs = self.organelles(a, m, c_in)
        
        if use_orchestrator:
            return self.orchestrator(raw_outputs)
        else:
            return raw_outputs
    
    def predict(self, a, m, c_in):
        """
        Predict result and flags.
        
        Returns:
            result: [batch] 0-255
            flags: dict with 'c', 'v', 'z', 'n' as [batch] 0/1
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(a, m, c_in)
            
            # Decode result
            result_probs = torch.sigmoid(outputs['result'])
            result = self.soroban.decode_batch(result_probs)
            
            # Decode flags
            c = (torch.sigmoid(outputs['c']) > 0.5).long().squeeze(-1)
            v = (torch.sigmoid(outputs['v']) > 0.5).long().squeeze(-1)
            nz_probs = torch.sigmoid(outputs['nz'])
            z = (nz_probs[:, 0] > 0.5).long()
            n = (nz_probs[:, 1] > 0.5).long()
            
            return result, {'c': c, 'v': v, 'z': z, 'n': n}
    
    def freeze_organelles(self):
        """Freeze organelle weights for Phase 2 training."""
        for param in self.organelles.parameters():
            param.requires_grad = False
    
    def unfreeze_organelles(self):
        """Unfreeze organelle weights."""
        for param in self.organelles.parameters():
            param.requires_grad = True
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def parameter_breakdown(self):
        """Return parameter count breakdown."""
        org_breakdown = self.organelles.parameter_breakdown()
        return {
            **org_breakdown,
            'orchestrator': self.orchestrator.num_parameters,
            'total': self.num_parameters,
        }
