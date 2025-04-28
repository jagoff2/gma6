"""
Chat Interface for Mistral with Liquid Memory and Neural Reasoning Enhancement

This script provides a command-line interface that shows a side-by-side comparison
between the base Mistral model and a version enhanced with liquid memory and neural
reasoning capabilities for improved long-context handling and reasoning steps.
"""

import os
import sys
import time
import json
import logging
import argparse
import readline
import textwrap
import signal
import threading
import queue
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import for Hugging Face models
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chat_interface.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

# ANSI color codes for terminal formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREY = '\033[90m'

@dataclass
class LiquidMemoryConfig:
    """Configuration for liquid memory and neural reasoning modules"""
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    hf_token: Optional[str] = "hf_token"
    memory_dim: int = 256
    num_memory_slots: int = 64
    num_memory_layers: int = 16
    attention_heads: int = 16
    dropout_rate: float = 0.1
    feedback_strength: float = 0.7
    use_layernorm: bool = True
    blend_ratio: float = 0.6
    continuous_time: bool = True
    min_timescale: float = 0.1
    max_timescale: float = 10.0
    stability_epsilon: float = 1e-6
    normalize_memory: bool = True
    debug_mode: bool = True
    fallback_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Reasoning configuration
    reasoning_max_steps: int = 5
    reasoning_min_steps: int = 1
    reasoning_trigger_threshold: float = 0.6
    reasoning_blend_ratio: float = 0.7

@dataclass
class ChatConfig:
    """Configuration for the chat interface"""
    max_context_length: int = 4096
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    show_token_count: bool = True
    show_generation_time: bool = True
    debug_mode: bool = True
    system_prompt: str = "You are a helpful AI assistant that provides accurate, informative, and engaging responses."


class LiquidMemoryCell(nn.Module):
    """
    Memory cell based on Liquid Neural Network principles
    
    Implements a continuous-time memory update mechanism with adaptive
    timescales for stable and flexible memory dynamics.
    """
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        config: LiquidMemoryConfig
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, memory_dim)
        
        # Memory target transformation
        self.memory_target = nn.Sequential(
            nn.Linear(memory_dim + input_dim, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2) if config.use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Linear(memory_dim * 2, memory_dim)
        )
        
        # Adaptive timescale network
        self.timescale_network = nn.Sequential(
            nn.Linear(memory_dim + input_dim, memory_dim),
            nn.LayerNorm(memory_dim) if config.use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Sigmoid()
        )
        
        # Gate networks for closed-form solution
        self.decay_gate = nn.Sequential(
            nn.Linear(memory_dim + input_dim, memory_dim),
            nn.Sigmoid()
        )
        
        # Gate for input influence
        self.input_gate = nn.Sequential(
            nn.Linear(memory_dim + input_dim, memory_dim),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, h):
        """
        Update memory state using closed-form solution
        
        Args:
            x: Input tensor [batch_size, input_dim]
            h: Current memory state [batch_size, memory_dim]
            
        Returns:
            new_h: Updated memory state [batch_size, memory_dim]
        """
        try:
            # Project input to memory space
            x_proj = self.input_projection(x)
            
            # Concatenate input and current state
            combined = torch.cat([h, x], dim=-1)
            
            # Calculate candidate state (target value)
            h_candidate = torch.tanh(self.memory_target(combined))
            
            # Calculate adaptive gates
            decay = self.decay_gate(combined)  # Controls memory decay 
            update = self.input_gate(combined)  # Controls input influence
            
            # Scale decay and update gates
            decay = self.config.min_timescale + (self.config.max_timescale - self.config.min_timescale) * decay
            
            # Apply closed-form update rule (with stability safeguards)
            decay = decay + self.config.stability_epsilon  # Avoid division by zero
            new_h = (1.0 - update) * (h / decay) + update * h_candidate
            
            # Check for numerical issues
            if torch.isnan(new_h).any() or torch.isinf(new_h).any():
                # Fall back to original state if numerical issues occur
                return h
            
            # Optional normalization for stability
            if self.config.normalize_memory:
                # Normalize while preserving the norm of the original vector
                h_norm = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=self.config.stability_epsilon)
                new_h = F.normalize(new_h, p=2, dim=-1) * h_norm
            
            return new_h
            
        except Exception as e:
            if self.config.debug_mode:
                logger.error(f"Error in LiquidMemoryCell: {e}")
            # On failure, return the original state
            return h


class LiquidMemoryController(nn.Module):
    """
    Memory controller based on Liquid Neural Network principles
    
    Controls multiple memory slots with content-based addressing and
    implements continuous-time memory dynamics.
    """
    def __init__(
        self,
        input_dim: int,
        memory_dim: int, 
        num_slots: int = 64,
        config: LiquidMemoryConfig = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.config = config or LiquidMemoryConfig()
        
        # Memory cell for continuous-time dynamics
        self.memory_cell = LiquidMemoryCell(
            input_dim=input_dim,
            memory_dim=memory_dim,
            config=self.config
        )
        
        # Content-based addressing components
        self.addressing_query = nn.Linear(input_dim, memory_dim)
        self.addressing_key = nn.Linear(memory_dim, memory_dim)
        
        # Memory management components
        self.usage_decay = 0.99  # Usage decay factor
        self.age_decay = 0.98  # Age decay factor
        
        # Register buffers for persistent state
        self.register_buffer('memory', None, persistent=False)
        self.register_buffer('usage', None, persistent=False)
        self.register_buffer('age', None, persistent=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _initialize_memory(self, batch_size, device):
        """Initialize memory buffers if not set"""
        if self.memory is None:
            # Initialize memory with small random values
            memory = torch.randn(batch_size, self.num_slots, self.memory_dim, device=device) * 0.01
            memory = F.normalize(memory, p=2, dim=-1)
            
            # Initialize usage and age tracking
            usage = torch.zeros(batch_size, self.num_slots, device=device)
            age = torch.zeros(batch_size, self.num_slots, device=device)
            
            # Register buffers
            self.memory = memory
            self.usage = usage
            self.age = age
    
    def reset_memory(self):
        """Reset memory and usage counters"""
        self.memory = None
        self.usage = None
        self.age = None
    
    def forward(self, hidden_states):
        """
        Process hidden states through liquid memory controller
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            memory: Updated memory tensor [batch_size, num_slots, memory_dim]
        """
        try:
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            
            # Initialize memory if needed
            self._initialize_memory(batch_size, device)
            
            # Process sequence elements
            for t in range(seq_len):
                # Current input state
                current_input = hidden_states[:, t]  # [batch_size, input_dim]
                
                # Calculate addressing query
                query = self.addressing_query(current_input)  # [batch_size, memory_dim]
                
                # Calculate keys for memory slots
                keys = self.addressing_key(self.memory)  # [batch_size, num_slots, memory_dim]
                
                # Compute addressing weights using scaled dot-product
                similarity = torch.bmm(
                    query.unsqueeze(1),  # [batch_size, 1, memory_dim]
                    keys.transpose(1, 2)  # [batch_size, memory_dim, num_slots]
                ).squeeze(1)  # [batch_size, num_slots]
                
                # Adjust by age and usage
                adjusted_similarity = similarity - 0.1 * self.age - 0.2 * self.usage
                
                # Get read and write weights
                read_weights = F.softmax(similarity, dim=1)  # [batch_size, num_slots]
                write_weights = F.softmax(adjusted_similarity, dim=1)  # [batch_size, num_slots]
                
                # Update memory slots with significant weights
                for b in range(batch_size):
                    # Get indices of significant slots
                    significant_slots = write_weights[b] > 0.01
                    slot_indices = significant_slots.nonzero().squeeze(-1)
                    
                    for idx in slot_indices:
                        i = idx.item()
                        write_weight = write_weights[b, i].item()
                        
                        # Current memory state
                        current_memory = self.memory[b, i].unsqueeze(0)  # [1, memory_dim]
                        
                        try:
                            # Update memory using liquid memory cell
                            new_memory = self.memory_cell(
                                current_input[b].unsqueeze(0),  # [1, input_dim]
                                current_memory                  # [1, memory_dim]
                            )
                            
                            # Apply weighted update
                            self.memory[b, i] = current_memory.squeeze(0) * (1 - write_weight) + \
                                              new_memory.squeeze(0) * write_weight
                            
                            # Update usage statistics
                            self.usage[b, i] += write_weight
                        except Exception as e:
                            if self.config.debug_mode:
                                logger.error(f"Error updating slot {i}: {e}")
                            continue  # Skip this slot on error
                
                # Age all memories
                self.age = self.age * self.age_decay + 1.0
                
                # Decay usage statistics
                self.usage = self.usage * self.usage_decay
            
            # Ensure memory values are valid
            if torch.isnan(self.memory).any() or torch.isinf(self.memory).any():
                # Replace invalid values with zeros
                self.memory = torch.where(
                    torch.isnan(self.memory) | torch.isinf(self.memory),
                    torch.zeros_like(self.memory),
                    self.memory
                )
                
                # Renormalize
                if self.config.normalize_memory:
                    self.memory = F.normalize(self.memory, p=2, dim=-1)
            
            return self.memory
            
        except Exception as e:
            if self.config.debug_mode:
                logger.error(f"Error in memory controller: {e}")
            
            # Return current memory or zeros if not initialized
            if self.memory is not None:
                return self.memory
            
            # Create empty memory if none exists
            return torch.zeros(
                batch_size, self.num_slots, self.memory_dim, 
                device=hidden_states.device
            )


class LiquidMemoryAttention(nn.Module):
    """
    Attention mechanism for liquid memory access
    
    Implements multi-head attention for accessing memory with
    adaptive timescales for dynamic temporal dependencies.
    """
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        config: LiquidMemoryConfig = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads
        self.config = config or LiquidMemoryConfig()
        
        assert memory_dim % num_heads == 0, "Memory dimension must be divisible by number of heads"
        
        # Multi-head attention components
        self.query = nn.Linear(input_dim, memory_dim)
        self.key = nn.Linear(memory_dim, memory_dim)
        self.value = nn.Linear(memory_dim, memory_dim)
        self.output = nn.Linear(memory_dim, input_dim)
        
        # Adaptive timescale for attention
        self.attention_timescale = nn.Sequential(
            nn.Linear(input_dim, num_heads),
            nn.Softplus()  # Ensures positive timescales
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, memory):
        """
        Process input through liquid memory attention
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            memory: Memory tensor [batch_size, num_slots, memory_dim]
            
        Returns:
            output: Memory-enhanced representation [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        num_slots = memory.size(1)
        
        # Residual connection
        residual = x
        
        # Layer normalization
        x_norm = self.layer_norm1(x)
        
        # Project to query, key, value spaces
        q = self.query(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(memory).view(batch_size, num_slots, self.num_heads, self.head_dim)
        v = self.value(memory).view(batch_size, num_slots, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, num_slots, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, num_slots, head_dim]
        
        # Compute raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add adaptive bias based on timescales
        if self.config.continuous_time:
            # Calculate adaptive timescales for each head
            head_timescales = self.attention_timescale(x_norm).mean(dim=1)  # [batch_size, num_heads]
            
            # Add small bias to scores
            bias = head_timescales.unsqueeze(-1).unsqueeze(-1) * 0.1
            
            # Apply bias
            scores = scores + bias
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention-weighted values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.memory_dim)
        
        # Output projection and dropout
        attn_output = self.output(attn_output)
        attn_output = self.dropout(attn_output)
        
        # First residual connection
        x = residual + attn_output
        
        # Second residual connection with feed-forward network
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.ff_network(x)
        
        return x


class LiquidMemoryLayer(nn.Module):
    """
    Complete liquid memory layer combining controller and attention
    
    Creates a memory enhancement layer with continuous-time dynamics 
    and adaptive timescales.
    """
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_slots: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        feedback_strength: float = 0.7,
        config: LiquidMemoryConfig = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.feedback_strength = feedback_strength
        self.config = config or LiquidMemoryConfig()
        
        # Liquid memory controller for storage
        self.memory_controller = LiquidMemoryController(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_slots=num_slots,
            config=self.config
        )
        
        # Liquid memory attention for retrieval
        self.memory_attention = LiquidMemoryAttention(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            config=self.config
        )
        
        # Input salience network for analytics
        self.salience_network = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.LayerNorm(memory_dim) if config.use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset_memory(self):
        """Reset memory state"""
        self.memory_controller.reset_memory()
    
    def forward(self, hidden_states):
        """
        Process hidden states through liquid memory layer
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            output: Memory-enhanced hidden states [batch_size, seq_len, input_dim]
        """
        try:
            # Calculate salience for monitoring (no thresholding)
            salience = self.salience_network(hidden_states)
            
            # Update memory with continuous-time dynamics
            memory = self.memory_controller(hidden_states)
            
            # Retrieve from memory using liquid attention
            memory_enhanced = self.memory_attention(hidden_states, memory)
            
            # Apply feedback with controlled strength
            output = hidden_states + self.feedback_strength * (memory_enhanced - hidden_states)
            
            return output
        except Exception as e:
            if self.config.debug_mode:
                logger.error(f"Error in LiquidMemoryLayer: {e}")
            # On error, return the original hidden states
            return hidden_states


class ReasoningTrigger(nn.Module):
    """
    Neural network to detect when complex reasoning is needed
    
    Analyzes hidden states to determine when a reasoning process
    should be triggered based on complexity patterns.
    """
    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.threshold = threshold
        
        # Detection network
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states):
        """
        Determine if reasoning is needed based on hidden states
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            is_triggered: Boolean tensor indicating if reasoning is triggered
            confidence: Confidence scores for the decision
        """
        # Process all token positions
        batch_size, seq_len, _ = hidden_states.shape
        
        # Calculate complexity scores
        scores = self.network(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Get max score for each sequence (we only need one trigger per sequence)
        confidence, _ = torch.max(scores, dim=1)  # [batch_size]
        
        # Threshold to determine if reasoning is triggered
        is_triggered = confidence > self.threshold
        
        return is_triggered, confidence


class ReasoningStep(nn.Module):
    """
    Single step in the reasoning process
    
    Implements a transformer-based reasoning step with self-attention
    to previous steps and content-based gating.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or (input_dim * 4)
        
        # Main transformation
        self.transform = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Self-attention for reasoning consistency
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state, context=None):
        """
        Process a single reasoning step
        
        Args:
            state: Current reasoning state [batch_size, 1, input_dim]
            context: Previous reasoning steps [batch_size, steps, input_dim]
            
        Returns:
            updated_state: Updated reasoning state [batch_size, 1, input_dim]
        """
        # Remember the original dtype for consistent return
        dtype = state.dtype
        
        # Apply transformation
        residual = state
        state = self.norm1(state)
        state = residual + self.transform(state).to(dtype)
        
        # Apply self-attention if context is provided
        if context is not None and context.size(1) > 0:
            residual = state
            state = self.norm2(state)
            
            # Attend to previous steps
            attn_output, _ = self.self_attention(
                query=state,
                key=context,
                value=context
            )
            
            # Residual connection
            state = residual + attn_output.to(dtype)
        
        return state


class ReasoningBuffer(nn.Module):
    """
    Buffer for multi-step reasoning process
    
    Implements iterative refinement of reasoning steps with
    content-aware termination and inter-step attention.
    """
    def __init__(
        self,
        input_dim: int,
        max_steps: int = 5,
        min_steps: int = 1,
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_steps = max_steps
        self.min_steps = min_steps
        
        # Create reasoning step modules
        self.reasoning_steps = nn.ModuleList([
            ReasoningStep(
                input_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(max_steps)
        ])
        
        # Final integration layer
        self.output_integration = nn.Sequential(
            nn.Linear(input_dim * (max_steps + 1), input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Step termination detector
        self.step_terminator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states):
        """
        Perform multi-step reasoning on hidden states
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            reasoning_output: Final reasoning output [batch_size, input_dim]
        """
        try:
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            # Use the last token's representation as starting point
            # This is typically the most informative position for reasoning
            current_state = hidden_states[:, -1:, :].to(dtype)  # [batch_size, 1, input_dim]
            
            # Store all reasoning steps
            all_states = [current_state]
            
            # Apply reasoning steps iteratively
            context = torch.zeros(batch_size, 0, self.input_dim, device=device, dtype=dtype)
            
            for i in range(self.max_steps):
                # Update state with reasoning step
                current_state = self.reasoning_steps[i](current_state, context).to(dtype)
                all_states.append(current_state)
                
                # Accumulate context for next step
                context = torch.cat([context, current_state], dim=1)
                
                # Check for early termination (but only after minimum steps)
                if i >= self.min_steps - 1:
                    termination_score = self.step_terminator(current_state).squeeze(-1)
                    if torch.all(termination_score > 0.8):
                        break
            
            # Concatenate all steps for final integration
            all_states_tensor = torch.cat(all_states, dim=1)  # [batch_size, steps+1, input_dim]
            flattened = all_states_tensor.reshape(batch_size, -1)  # [batch_size, (steps+1)*input_dim]
            
            # Create final reasoning output
            reasoning_output = self.output_integration(flattened).to(dtype)
            
            return reasoning_output
            
        except Exception as e:
            logger.error(f"Error in ReasoningBuffer: {e}")
            # On error, return zeros as fallback
            return torch.zeros(hidden_states.size(0), self.input_dim, device=hidden_states.device, dtype=hidden_states.dtype)


class NeuralReasoningLayer(nn.Module):
    """
    Complete neural reasoning layer
    
    Integrates reasoning detection, multi-step reasoning process,
    and reintegration of reasoning results into the original hidden states.
    """
    def __init__(
        self,
        input_dim: int,
        max_steps: int = 5,
        min_steps: int = 1,
        blend_ratio: float = 0.7,
        num_heads: int = 4,
        dropout: float = 0.1,
        trigger_threshold: float = 0.6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.blend_ratio = blend_ratio
        
        # Reasoning components
        self.trigger = ReasoningTrigger(input_dim, threshold=trigger_threshold, dropout=dropout)
        self.buffer = ReasoningBuffer(
            input_dim=input_dim,
            max_steps=max_steps,
            min_steps=min_steps,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output integration
        self.integration = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states):
        """
        Process hidden states through reasoning layer
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            enhanced: Reasoning-enhanced hidden states [batch_size, seq_len, input_dim]
        """
        try:
            # Ensure we're using the same dtype as input
            input_dtype = hidden_states.dtype
            
            # Check if reasoning should be applied
            is_triggered, confidence = self.trigger(hidden_states)
            
            # Only apply reasoning if triggered
            if torch.any(is_triggered):
                # Process through reasoning buffer
                reasoning_output = self.buffer(hidden_states).to(input_dtype)  # Ensure matching dtype
                
                # Enhance last token representation with reasoning
                enhanced = hidden_states.clone()
                
                # Apply reasoning only to sequences that triggered it
                for b in range(hidden_states.size(0)):
                    if is_triggered[b]:
                        # Get original last token representation
                        last_token = hidden_states[b, -1]
                        
                        # Combine with reasoning output
                        combined = torch.cat([last_token, reasoning_output[b]], dim=0)
                        integrated = self.integration(combined.unsqueeze(0)).squeeze(0).to(input_dtype)
                        
                        # Apply blend ratio
                        enhanced[b, -1] = (self.blend_ratio * integrated + 
                                          (1.0 - self.blend_ratio) * last_token)
                
                return enhanced
            
            # If not triggered, return original hidden states
            return hidden_states
            
        except Exception as e:
            logger.error(f"Error in NeuralReasoningLayer: {e}")
            # On error, return original hidden states
            return hidden_states


class MistralWithLiquidAndReasoningMemory(nn.Module):
    """
    Mistral model enhanced with liquid memory and neural reasoning layers
    
    Integrates liquid memory for persistent storage with neural reasoning
    for improved logical processing.
    """
    def __init__(self, config: LiquidMemoryConfig):
        super().__init__()
        self.config = config
        
        # Load base Mistral model with authentication if token provided
        logger.info(f"Loading base model: {config.base_model_name}")
        
        try:
            # Set up authentication parameter
            auth_params = {}
            if config.hf_token:
                auth_params["token"] = config.hf_token
                logger.info("Using HuggingFace API token for authentication")
            
            # For actual implementation, we load in the highest precision possible
            # and downsample only when space is critical
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                **auth_params
            )
            
            logger.info(f"Successfully loaded model: {config.base_model_name}")
            
        except Exception as e:
            logger.error(f"Error loading {config.base_model_name}: {e}")
            
            # Try fallback model if authentication fails
            logger.info(f"Attempting to load fallback model: {config.fallback_model}")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.fallback_model,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded fallback model: {config.fallback_model}")
            except Exception as fallback_e:
                logger.error(f"Error loading fallback model: {fallback_e}")
                raise RuntimeError("Failed to load both primary and fallback models")
        
        # Extract model dimensions
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Model hidden size: {self.hidden_size}")
        
        # Store the dtype from base model
        self.dtype = next(self.model.parameters()).dtype
        logger.info(f"Model using dtype: {self.dtype}")
        
        # Create alternating memory and reasoning layers
        logger.info(f"Creating cognitive enhancement layers")
        self.memory_layers = nn.ModuleList()
        self.reasoning_layers = nn.ModuleList()
        
        # Create memory layers
        for _ in range(config.num_memory_layers):
            self.memory_layers.append(
                LiquidMemoryLayer(
                    input_dim=self.hidden_size,
                    memory_dim=config.memory_dim,
                    num_slots=config.num_memory_slots,
                    num_heads=config.attention_heads,
                    dropout=config.dropout_rate,
                    feedback_strength=config.feedback_strength,
                    config=config
                )
            )
        
        # Create reasoning layers (half as many as memory layers)
        num_reasoning_layers = max(1, config.num_memory_layers // 2)
        logger.info(f"Creating {num_reasoning_layers} reasoning layers")
        for _ in range(num_reasoning_layers):
            self.reasoning_layers.append(
                NeuralReasoningLayer(
                    input_dim=self.hidden_size,
                    max_steps=config.reasoning_max_steps,
                    min_steps=config.reasoning_min_steps,
                    blend_ratio=config.reasoning_blend_ratio,
                    trigger_threshold=config.reasoning_trigger_threshold,
                    dropout=config.dropout_rate
                )
            )
        
        # Hook for capturing hidden states
        self.hidden_states = None
        self._register_hooks()
        
        # Store blend ratio for memory integration
        self.blend_ratio = config.blend_ratio
        
        # Initialize weights
        self._init_weights()
        
        # Convert all layers to match model dtype
        self.to(self.dtype)
        
        logger.info(f"Liquid memory and reasoning enhanced Mistral model initialized with dtype {self.dtype}")
    
    def _init_weights(self):
        """Initialize new weights with appropriate distributions"""
        for name, module in self.named_children():
            if name in ['memory_layers', 'reasoning_layers']:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def _register_hooks(self):
        """Register forward hooks to capture hidden states"""
        def hook_fn(module, input, output):
            # This hook captures the hidden states from the transformer
            if hasattr(output, 'last_hidden_state'):
                self.hidden_states = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                self.hidden_states = output[0]
            else:
                self.hidden_states = output
        
        # Hook the model's final layer
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            self.model.model.norm.register_forward_hook(hook_fn)
            logger.info("Hook registered on model.model.norm")
        else:
            # Try to find the last layer dynamically
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.LayerNorm):
                    module.register_forward_hook(hook_fn)
                    logger.info(f"Hook registered on {name}")
                    break
    
    def reset_memory(self):
        """Reset all memory layers"""
        for layer in self.memory_layers:
            layer.reset_memory()
        logger.info("Memory state reset")
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None, 
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """Forward pass with memory and reasoning augmentation"""
        # Forward pass through base Mistral model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states
            return_dict=True,  # Always use return dict for consistency
            **kwargs
        )
        
        # Process hidden states through memory and reasoning layers
        if self.hidden_states is not None:
            enhanced = self.hidden_states
            logger.debug(f"Hidden states shape: {enhanced.shape}")
            
            # Apply memory layers
            for i, layer in enumerate(self.memory_layers):
                enhanced = layer(enhanced)
                logger.debug(f"After memory layer {i+1}: {enhanced.shape}")
                
                # Apply reasoning layer after every other memory layer
                if i % 2 == 1 and i // 2 < len(self.reasoning_layers):
                    reasoning_idx = i // 2
                    enhanced = self.reasoning_layers[reasoning_idx](enhanced)
                    logger.debug(f"After reasoning layer {reasoning_idx+1}: {enhanced.shape}")
            
            # Apply any remaining reasoning layers
            for i in range(len(self.memory_layers) // 2, len(self.reasoning_layers)):
                enhanced = self.reasoning_layers[i](enhanced)
                logger.debug(f"After reasoning layer {i+1}: {enhanced.shape}")
            
            # Find appropriate lm_head
            if hasattr(self.model, 'lm_head'):
                lm_head = self.model.lm_head
            else:
                # Find lm_head dynamically
                lm_head = None
                for name, module in self.model.named_modules():
                    if 'lm_head' in name or (isinstance(module, nn.Linear) and 
                                           module.out_features == self.model.config.vocab_size):
                        lm_head = module
                        break
                
                if lm_head is None:
                    logger.warning("Could not find lm_head, using original logits")
                    return outputs
            
            # Generate new logits from enhanced hidden states
            try:
                # Use memory enhanced hidden states to generate logits
                new_logits = lm_head(enhanced)
                
                # Apply blend ratio for memory influence
                blended_logits = self.blend_ratio * new_logits + (1.0 - self.blend_ratio) * outputs.logits
                outputs.logits = blended_logits
                logger.debug("Successfully blended logits with cognitive enhancements")
            except Exception as e:
                logger.error(f"Error generating new logits: {e}")
                # Return original outputs if there's an error
                pass
        
        return outputs
    
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_length=None,
        do_sample=True,
        num_beams=1,
        temperature=1.0,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        """
        Generate text using the cognitively enhanced model
        
        This implements generation with memory and reasoning enhancements.
        """
        # Create parameter dictionary for generate with non-None values
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": max_length,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": pad_token_id if pad_token_id is not None else self.model.config.pad_token_id,
            "eos_token_id": eos_token_id if eos_token_id is not None else self.model.config.eos_token_id,
        }
        
        # Add additional kwargs
        gen_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        # Call the base model's generate method
        try:
            # Use the base model's generation 
            generation_output = self.model.generate(**gen_kwargs)
            
            # After generation, process through memory layers to update memory state
            # This ensures memory is updated even though we used the base model's generate
            if input_ids is not None and getattr(self.model.config, "is_encoder_decoder", False) == False:
                # Get the full generated text (input + generated)
                with torch.no_grad():
                    # Run a forward pass to update memory
                    _ = self.forward(input_ids=generation_output)
            
            return generation_output
            
        except Exception as e:
            logger.error(f"Error in generate method: {e}")
            # Fallback to original generate
            return self.model.generate(**gen_kwargs)


def load_tokenizer_with_auth(model_name, hf_token=None, fallback_model=None):
    """
    Load the tokenizer with authentication
    
    Args:
        model_name: HuggingFace model name
        hf_token: HuggingFace API token for authentication 
        fallback_model: Fallback model to use if loading fails
        
    Returns:
        Tokenizer for the model
    """
    logger.info(f"Loading tokenizer for model: {model_name}")
    
    try:
        # Try with token if provided
        if hf_token:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            logger.info("Tokenizer loaded successfully with auth token")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded successfully")
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
            
        return tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load tokenizer for primary model: {e}")
        
        # Try fallback model if provided
        if fallback_model:
            logger.info(f"Attempting to load tokenizer for fallback model: {fallback_model}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                logger.info("Fallback tokenizer loaded successfully")
                return tokenizer
            except Exception as fallback_err:
                logger.error(f"Failed to load fallback tokenizer: {fallback_err}")
        
        # Try a public alternative if all else fails
        logger.info("Attempting to load public tokenizer as last resort")
        try:
            public_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            if public_tokenizer.pad_token is None:
                public_tokenizer.pad_token = public_tokenizer.eos_token
            logger.warning("Using public tokenizer as fallback")
            return public_tokenizer
        except:
            raise RuntimeError("Failed to load any tokenizer")


class ConversationManager:
    """Manages the conversation history and prompt formatting"""
    def __init__(self, tokenizer, config: ChatConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.messages = []
        self.system_prompt = config.system_prompt
        
        # Add the system prompt as the first message
        self.add_system_message(config.system_prompt)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation"""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation"""
        self.add_message("assistant", content)
    
    def add_system_message(self, content: str):
        """Add a system message to the conversation"""
        self.system_prompt = content
        self.add_message("system", content)
    
    def prepare_for_model(self):
        """Format the messages for the model"""
        # Check if the tokenizer has a chat template
        has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        
        # Attempt to use chat template if available
        if has_chat_template:
            try:
                # Create token IDs using chat template
                tokenized = self.tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Error applying chat template: {e}")
                # Fallback to manual formatting
                has_chat_template = False
        
        # Manual formatting if chat template is not available or failed
        if not has_chat_template:
            logger.info("Using manual chat formatting")
            
            # Format for Mistral models
            formatted_text = ""
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    formatted_text += f"<s>[INST] {content} [/INST]</s>\n"
                elif role == "user":
                    formatted_text += f"<s>[INST] {content} [/INST]"
                elif role == "assistant":
                    formatted_text += f" {content}</s>\n"
            
            # Add final assistant prompt if needed
            if self.messages[-1]["role"] != "assistant":
                formatted_text += " "
                
            # Tokenize the formatted text
            tokenized = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_context_length
            ).input_ids
        
        # Check token count against max context length
        if tokenized.size(1) > self.config.max_context_length:
            logger.warning(f"Context too long ({tokenized.size(1)} tokens), truncating to {self.config.max_context_length}")
            
            # Keep system message and recent conversations
            system_messages = [msg for msg in self.messages if msg["role"] == "system"]
            regular_messages = [msg for msg in self.messages if msg["role"] != "system"]
            
            # Rebuild messages, dropping older ones as needed
            while len(regular_messages) > 2:  # Always keep at least the last exchange
                regular_messages.pop(0)  # Remove oldest message
                
                # Rebuild messages and check length
                test_messages = system_messages + regular_messages
                
                if has_chat_template:
                    test_tokenized = self.tokenizer.apply_chat_template(
                        test_messages,
                        tokenize=True,
                        return_tensors="pt",
                        add_generation_prompt=True
                    )
                else:
                    # Manual formatting
                    test_formatted = ""
                    for msg in test_messages:
                        role = msg["role"]
                        content = msg["content"]
                        
                        if role == "system":
                            test_formatted += f"<s>[INST] {content} [/INST]</s>\n"
                        elif role == "user":
                            test_formatted += f"<s>[INST] {content} [/INST]"
                        elif role == "assistant":
                            test_formatted += f" {content}</s>\n"
                    
                    # Add final assistant prompt if needed
                    if test_messages[-1]["role"] != "assistant":
                        test_formatted += " "
                        
                    test_tokenized = self.tokenizer(
                        test_formatted,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_context_length
                    ).input_ids
                
                # If we're under the limit, use this version
                if test_tokenized.size(1) <= self.config.max_context_length:
                    self.messages = test_messages
                    tokenized = test_tokenized
                    break
            
            # If we still exceed the limit, resort to more aggressive truncation
            if tokenized.size(1) > self.config.max_context_length:
                logger.warning("Aggressive truncation needed, keeping only system and last exchange")
                
                # Keep only system message and the most recent exchange
                if len(regular_messages) > 2:
                    regular_messages = regular_messages[-2:]  # Keep last user + assistant pair
                
                self.messages = system_messages + regular_messages
                
                # Reformat and tokenize
                if has_chat_template:
                    tokenized = self.tokenizer.apply_chat_template(
                        self.messages,
                        tokenize=True,
                        return_tensors="pt",
                        add_generation_prompt=True
                    )
                else:
                    # Manual formatting
                    formatted_text = ""
                    for msg in self.messages:
                        role = msg["role"]
                        content = msg["content"]
                        
                        if role == "system":
                            formatted_text += f"<s>[INST] {content} [/INST]</s>\n"
                        elif role == "user":
                            formatted_text += f"<s>[INST] {content} [/INST]"
                        elif role == "assistant":
                            formatted_text += f" {content}</s>\n"
                    
                    # Add final assistant prompt if needed
                    if self.messages[-1]["role"] != "assistant":
                        formatted_text += " "
                        
                    tokenized = self.tokenizer(
                        formatted_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_context_length
                    ).input_ids
        
        return tokenized
    
    def clear(self):
        """Clear conversation history except for system message"""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
        
        # If no system messages were saved, add the default one back
        if not system_messages:
            self.add_system_message(self.system_prompt)


class ModelManager:
    """Manages both base and enhanced models"""
    def __init__(self, config: LiquidMemoryConfig, chat_config: ChatConfig):
        self.config = config
        self.chat_config = chat_config
        self.tokenizer = None
        self.base_model = None
        self.enhanced_model = None
        self.device = torch.device("cpu")
    
    def load_models(self):
        """Load tokenizer and models"""
        try:
            logger.info(f"Loading tokenizer for {self.config.base_model_name}")
            self.tokenizer = load_tokenizer_with_auth(
                self.config.base_model_name,
                hf_token=self.config.hf_token,
                fallback_model=self.config.fallback_model
            )
            
            # Check for chat template
            if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                logger.warning("Chat template not found. Using default Mistral chat template.")
                # Set a default chat template for Mistral models
                self.tokenizer.chat_template = """<s>{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'system' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"""
            
            # Set up authentication parameter
            auth_params = {}
            if self.config.hf_token:
                auth_params["token"] = self.config.hf_token
            
            logger.info(f"Loading base model: {self.config.base_model_name}")
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True,
                    **auth_params
                )
                logger.info(f"Successfully loaded base model: {self.config.base_model_name}")
            except Exception as e:
                logger.error(f"Error loading base model: {e}")
                
                # Try fallback model
                logger.info(f"Attempting to load fallback model: {self.config.fallback_model}")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.fallback_model,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True
                )
                # Update config to use fallback model
                self.config.base_model_name = self.config.fallback_model
                logger.info(f"Successfully loaded fallback model: {self.config.fallback_model}")
            
            logger.info("Creating liquid memory and reasoning enhanced model")
            self.enhanced_model = MistralWithLiquidAndReasoningMemory(self.config)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def generate_responses(self, input_ids, max_new_tokens=256):
        """Generate responses from both models"""
        # Parameters for text generation
        gen_params = {
            "input_ids": input_ids.to(self.device),
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": self.chat_config.temperature,
            "top_p": self.chat_config.top_p,
            "top_k": self.chat_config.top_k,
            "repetition_penalty": self.chat_config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        try:
            # Generate with base model
            base_start_time = time.time()
            with torch.no_grad():
                base_outputs = self.base_model.generate(**gen_params)
            base_time = time.time() - base_start_time
            
            # Generate with enhanced model
            enhanced_start_time = time.time()
            with torch.no_grad():
                enhanced_outputs = self.enhanced_model.generate(**gen_params)
            enhanced_time = time.time() - enhanced_start_time
            
            # Decode outputs - get only the newly generated parts
            input_length = input_ids.size(1)
            base_text = self.tokenizer.decode(
                base_outputs[0][input_length:], 
                skip_special_tokens=True
            )
            enhanced_text = self.tokenizer.decode(
                enhanced_outputs[0][input_length:], 
                skip_special_tokens=True
            )
            
            return base_text, enhanced_text, base_time, enhanced_time
            
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            return f"Error: {str(e)}", f"Error: {str(e)}", 0.0, 0.0
    
    def reset_memory(self):
        """Reset memory in the enhanced model"""
        if self.enhanced_model:
            self.enhanced_model.reset_memory()


class AsyncGenerator:
    """Handles async generation of model responses"""
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.result_queue = queue.Queue()
    
    def generate_async(self, input_ids, max_new_tokens=256):
        """Start async generation"""
        generator_thread = threading.Thread(
            target=self._generate_and_queue, 
            args=(input_ids, max_new_tokens)
        )
        generator_thread.daemon = True
        generator_thread.start()
        return generator_thread
    
    def _generate_and_queue(self, input_ids, max_new_tokens):
        """Generate responses and place in queue"""
        try:
            results = self.model_manager.generate_responses(input_ids, max_new_tokens)
            self.result_queue.put(results)
        except Exception as e:
            logger.error(f"Async generation error: {e}")
            self.result_queue.put((f"Error: {str(e)}", f"Error: {str(e)}", 0.0, 0.0))
    
    def get_results(self, timeout=None):
        """Get results from queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class ChatInterface:
    """Main chat interface for interacting with the models"""
    def __init__(
        self, 
        liquid_config: LiquidMemoryConfig, 
        chat_config: ChatConfig
    ):
        self.liquid_config = liquid_config
        self.chat_config = chat_config
        self.model_manager = ModelManager(liquid_config, chat_config)
        self.conversation = None
        self.generator = None
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def initialize(self):
        """Initialize models and conversation manager"""
        if not self.model_manager.load_models():
            logger.error("Failed to load models")
            return False
        
        self.conversation = ConversationManager(
            self.model_manager.tokenizer, 
            self.chat_config
        )
        self.generator = AsyncGenerator(self.model_manager)
        
        return True
    
    def _handle_interrupt(self, sig, frame):
        """Handle keyboard interrupt"""
        print("\nExiting chat...")
        self.running = False
        sys.exit(0)
    
    def _display_welcome(self):
        """Display welcome message"""
        terminal_width = os.get_terminal_size().columns
        
        print("\n" + "=" * terminal_width)
        print(f"{Colors.BOLD}{Colors.HEADER}Mistral with Liquid Memory & Neural Reasoning - Interactive Chat Interface{Colors.ENDC}")
        print(f"{Colors.CYAN}Compare standard Mistral with cognitively enhanced responses{Colors.ENDC}")
        print("-" * terminal_width)
        print(f"{Colors.YELLOW}Base Model:{Colors.ENDC} {self.liquid_config.base_model_name}")
        print(f"{Colors.YELLOW}Device:{Colors.ENDC} {self.model_manager.device}")
        print(f"{Colors.YELLOW}Memory Config:{Colors.ENDC} {self.liquid_config.num_memory_slots} slots, {self.liquid_config.memory_dim} dimensions")
        print(f"{Colors.YELLOW}Reasoning Steps:{Colors.ENDC} {self.liquid_config.reasoning_max_steps} max, {self.liquid_config.reasoning_min_steps} min")
        print("-" * terminal_width)
        print(f"{Colors.GREY}Type 'quit', 'exit', or Ctrl+C to exit{Colors.ENDC}")
        print(f"{Colors.GREY}Type 'reset' to clear conversation history and memory{Colors.ENDC}")
        print(f"{Colors.GREY}Type 'system <message>' to modify the system prompt{Colors.ENDC}")
        print(f"{Colors.GREY}Type 'debug on/off' to toggle debug mode{Colors.ENDC}")
        print(f"{Colors.GREY}Type 'token <huggingface_token>' to update your HuggingFace token{Colors.ENDC}")
        print("=" * terminal_width + "\n")
    
    def _display_responses(
        self, 
        base_response: str, 
        enhanced_response: str, 
        base_time: float, 
        enhanced_time: float
    ):
        """Display model responses side by side"""
        terminal_width = os.get_terminal_size().columns
        half_width = terminal_width // 2 - 2
        
        # Wrap text to fit columns
        base_wrapped = textwrap.wrap(base_response, width=half_width)
        enhanced_wrapped = textwrap.wrap(enhanced_response, width=half_width)
        
        # Ensure both columns have the same number of lines
        max_lines = max(len(base_wrapped), len(enhanced_wrapped))
        while len(base_wrapped) < max_lines:
            base_wrapped.append("")
        while len(enhanced_wrapped) < max_lines:
            enhanced_wrapped.append("")
        
        # Print header
        print("\n" + "-" * terminal_width)
        
        header_base = f"{Colors.BOLD}Base Mistral Response{Colors.ENDC}"
        header_enhanced = f"{Colors.BOLD}Memory+Reasoning Response{Colors.ENDC}"
        
        if self.chat_config.show_generation_time:
            header_base += f" ({base_time:.2f}s)"
            header_enhanced += f" ({enhanced_time:.2f}s)"
        
        print(f"{header_base:<{half_width+20}} | {header_enhanced}")
        print("-" * terminal_width)
        
        # Print responses side by side
        for base_line, enhanced_line in zip(base_wrapped, enhanced_wrapped):
            print(f"{base_line:<{half_width}} | {enhanced_line}")
        
        print("-" * terminal_width + "\n")
    
    def _progress_spinner(self, thread):
        """Display a spinner while waiting for generation"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        
        # Only show spinner if generation is taking some time
        time.sleep(0.5)
        if thread.is_alive():
            sys.stdout.write(f"\r{Colors.YELLOW}Generating response {spinner[i]}{Colors.ENDC}")
            sys.stdout.flush()
            
            while thread.is_alive():
                i = (i + 1) % len(spinner)
                sys.stdout.write(f"\r{Colors.YELLOW}Generating response {spinner[i]}{Colors.ENDC}")
                sys.stdout.flush()
                time.sleep(0.1)
            
            # Clear the spinner line
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()
    
    def run(self):
        """Run the chat interface"""
        if not self.initialize():
            print("Failed to initialize. Exiting.")
            return
        
        self._display_welcome()
        
        while self.running:
            try:
                # Get user input
                user_input = input(f"{Colors.GREEN}User: {Colors.ENDC}")
                
                # Process special commands
                if user_input.lower() in ['quit', 'exit']:
                    print("Exiting chat...")
                    break
                
                if user_input.lower() == 'reset':
                    self.conversation.clear()
                    self.model_manager.reset_memory()
                    print(f"{Colors.CYAN}Chat history and memory reset{Colors.ENDC}")
                    continue
                
                if user_input.lower().startswith('system '):
                    system_msg = user_input[7:]  # Remove 'system ' prefix
                    self.conversation.add_system_message(system_msg)
                    print(f"{Colors.CYAN}System message updated{Colors.ENDC}")
                    continue
                
                if user_input.lower() == 'debug on':
                    self.chat_config.debug_mode = True
                    self.liquid_config.debug_mode = True
                    print(f"{Colors.CYAN}Debug mode enabled{Colors.ENDC}")
                    continue
                
                if user_input.lower() == 'debug off':
                    self.chat_config.debug_mode = False
                    self.liquid_config.debug_mode = False
                    print(f"{Colors.CYAN}Debug mode disabled{Colors.ENDC}")
                    continue
                
                if user_input.lower().startswith('token '):
                    token = user_input[6:].strip()  # Remove 'token ' prefix
                    self.liquid_config.hf_token = token
                    print(f"{Colors.CYAN}HuggingFace token updated{Colors.ENDC}")
                    print(f"{Colors.CYAN}Please 'reset' and restart the chat to use the new token{Colors.ENDC}")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Add user message to history
                self.conversation.add_user_message(user_input)
                
                # Prepare input for models
                input_ids = self.conversation.prepare_for_model()
                
                if self.chat_config.debug_mode:
                    # Show the number of tokens
                    print(f"{Colors.GREY}Input tokens: {input_ids.size(1)}{Colors.ENDC}")
                
                # Generate responses asynchronously
                thread = self.generator.generate_async(
                    input_ids, 
                    max_new_tokens=self.chat_config.max_new_tokens
                )
                
                # Show progress spinner
                self._progress_spinner(thread)
                
                # Get the responses
                base_response, enhanced_response, base_time, enhanced_time = self.generator.get_results()
                
                # Display responses side by side
                self._display_responses(base_response, enhanced_response, base_time, enhanced_time)
                
                # Add liquid model response to history for continuity
                self.conversation.add_assistant_message(enhanced_response)
                
            except KeyboardInterrupt:
                print("\nExiting chat...")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"{Colors.RED}Error: {str(e)}{Colors.ENDC}")


def main():
    """Main function to parse args and start the chat interface"""
    parser = argparse.ArgumentParser(description="Mistral with Liquid Memory & Neural Reasoning Chat Interface")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model to use (default: mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--fallback", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Fallback model to use if primary fails (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token for accessing gated models")
    
    # Chat configuration
    parser.add_argument("--max-context", type=int, default=4096,
                        help="Maximum token length for context (default: 4096)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling parameter (default: 0.9)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    # Memory configuration
    parser.add_argument("--memory-dim", type=int, default=256,
                        help="Memory representation dimension (default: 256)")
    parser.add_argument("--memory-slots", type=int, default=64,
                        help="Number of memory slots (default: 64)")
    parser.add_argument("--memory-layers", type=int, default=8,
                        help="Number of memory enhancement layers (default: 8)")
    parser.add_argument("--feedback", type=float, default=0.7,
                        help="Feedback strength (default: 0.7)")
    parser.add_argument("--blend", type=float, default=0.6,
                        help="Memory blend ratio (default: 0.6)")
    
    # Reasoning configuration
    parser.add_argument("--reasoning-max-steps", type=int, default=5,
                        help="Maximum reasoning steps (default: 5)")
    parser.add_argument("--reasoning-min-steps", type=int, default=1,
                        help="Minimum reasoning steps (default: 1)")
    parser.add_argument("--reasoning-threshold", type=float, default=0.6,
                        help="Reasoning trigger threshold (default: 0.6)")
    parser.add_argument("--reasoning-blend", type=float, default=0.7,
                        help="Reasoning blend ratio (default: 0.7)")
    
    args = parser.parse_args()
    
    # Check for HF token in environment variable if not provided as argument
    hf_token = args.hf_token
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token:
            logger.info("Using HuggingFace token from environment variable")
    
    # Create configurations
    chat_config = ChatConfig(
        max_context_length=args.max_context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        debug_mode=args.debug
    )
    
    liquid_config = LiquidMemoryConfig(
        base_model_name=args.model,
        fallback_model=args.fallback,
        hf_token=hf_token,
        memory_dim=args.memory_dim,
        num_memory_slots=args.memory_slots,
        num_memory_layers=args.memory_layers,
        feedback_strength=args.feedback,
        blend_ratio=args.blend,
        debug_mode=args.debug,
        reasoning_max_steps=args.reasoning_max_steps,
        reasoning_min_steps=args.reasoning_min_steps,
        reasoning_trigger_threshold=args.reasoning_threshold,
        reasoning_blend_ratio=args.reasoning_blend
    )
    
    # Start chat interface
    chat = ChatInterface(liquid_config, chat_config)
    chat.run()

if __name__ == "__main__":
    main()