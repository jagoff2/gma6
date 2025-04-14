import os
import json
import math
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import colorama
from colorama import Fore, Style
import getpass
from huggingface_hub import login
from transformers import StoppingCriteria, StoppingCriteriaList

class AssistantEndTokenCriteria(StoppingCriteria):
    """Custom stopping criteria to stop generation when </assistant> token sequence is generated"""
    def __init__(self, tokenizer, device):
        self.end_token_ids = tokenizer.encode("</assistant>", add_special_tokens=False)
        self.device = device
        self.sequence_length = len(self.end_token_ids)
    
    def __call__(self, input_ids, scores, **kwargs):
        # Don't check if we haven't generated enough tokens yet
        if input_ids.shape[1] < self.sequence_length:
            return False
        
        # Check each sequence in the batch
        for seq_idx in range(input_ids.shape[0]):
            num_windows = input_ids.shape[1] - self.sequence_length + 1
            
            for window_start in range(num_windows):
                window_end = window_start + self.sequence_length
                window = input_ids[seq_idx, window_start:window_end]
                
                # Check if this window matches the end token sequence
                if torch.equal(window, torch.tensor(self.end_token_ids, device=self.device)):
                    return True
        
        return False



# Initialize colorama
colorama.init()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
# Configuration Classes
#-------------------------------------------------------------------------------



@dataclass
class MemoryConfig:
    """Configuration for memory-enhanced model"""
    base_model_name: str = "google/gemma-3-27b-it"
    hf_token: Optional[str] = None
    memory_dim: int = 5376  # Updated to match the model's dimension from error
    num_memory_slots: int = 5376  # Reduced for CPU efficiency
    memory_update_rate: float = 0.5
    num_memory_layers: int = 8  # Reduced for CPU efficiency
    attention_heads: int = 16  # Increased for compatibility with larger dimension
    dropout_rate: float = 0.0
    use_gru_controller: bool = True
    feedback_strength: float = 0.8
    blend_ratio: float = 0.7
    offload_to_cpu: bool = True
    max_memory_usage: int = 64
    use_half_precision: bool = True
    use_4bit_quantization: bool = False
    gradient_checkpointing: bool = False
#-------------------------------------------------------------------------------
# Memory Module Components
#-------------------------------------------------------------------------------

class MemoryController(nn.Module):
    """Memory controller for storing and retrieving information with debug instrumentation"""
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_slots: int = 16,
        update_rate: float = 0.5,
        use_gru: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.update_rate = update_rate
        self.use_gru = use_gru
        
        logger.info(f"MemoryController initialized with input_dim={input_dim}, memory_dim={memory_dim}, num_slots={num_slots}")
        
        # Memory content projections
        self.input_transform = nn.Linear(input_dim, memory_dim)
        self.key_transform = nn.Linear(memory_dim, memory_dim)
        self.value_transform = nn.Linear(input_dim, memory_dim)
        self.query_transform = nn.Linear(input_dim, memory_dim)
        
        # Memory state controller
        if use_gru:
            self.memory_gate = nn.Linear(memory_dim * 2, memory_dim)
            self.memory_update = nn.Linear(memory_dim * 2, memory_dim)
            self.memory_reset = nn.Linear(memory_dim * 2, memory_dim)
        else:
            self.memory_update = nn.Linear(memory_dim * 2, memory_dim)
            self.memory_gate = nn.Linear(memory_dim * 2, memory_dim)
        
        # For memory access
        self.content_score = nn.Linear(memory_dim, 1)
            
        self.age_factor = 0.98  # Decay factor for memory age
        
        # Register buffers to properly handle state
        self.register_buffer('memory', None, persistent=True)
        self.register_buffer('usage', None, persistent=True)
        self.register_buffer('age', None, persistent=True)
        
        # Debugging ID
        self.debug_layer_id = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def set_debug_id(self, layer_id):
        """Set a layer identifier for debug messages"""
        self.debug_layer_id = layer_id
    
    def log_debug(self, message):
        """Log debug message with layer identifier"""
        prefix = f"[Memory Controller {self.debug_layer_id}] " if self.debug_layer_id is not None else "[Memory Controller] "
        logger.info(f"{prefix}{message}")
    
    def reset_memory(self):
        """Reset memory and usage counters"""
        self.memory = None
        self.usage = None
        self.age = None
        self.log_debug("Memory state reset")
    
    def _initialize_memory(self, batch_size, device, dtype=torch.float32):
        """Initialize memory if not already set"""
        if self.memory is None or self.memory.size(0) != batch_size:
            self.log_debug(f"Initializing memory with batch_size={batch_size}, num_slots={self.num_slots}, memory_dim={self.memory_dim}")
            
            # Initialize with small random values
            memory = torch.randn(batch_size, self.num_slots, self.memory_dim, 
                              device=device, dtype=dtype) * 0.01
            memory = F.normalize(memory, p=2, dim=-1)  # Initialize on unit hypersphere
            
            # Initialize usage as all zeros (no usage)
            usage = torch.zeros(batch_size, self.num_slots, device=device, dtype=dtype)
            
            # Initialize age tracking
            age = torch.zeros(batch_size, self.num_slots, device=device, dtype=dtype)
            
            # Set buffers
            self.memory = memory
            self.usage = usage
            self.age = age
            
            self.log_debug(f"Memory initialized with shape: {memory.shape}, dtype: {memory.dtype}")
            
            # Sample first few memory entries for verification
            self.log_debug(f"Memory sample (first entry): {memory[0, 0, :5].tolist()}")
    
    def forward(self, hidden_states):
        """Process hidden states through memory controller with debug instrumentation"""
        try:
            self.log_debug(f"Forward called with hidden_states shape: {hidden_states.shape}")
            
            # Handle input type and device
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            self.log_debug(f"Detected batch_size={batch_size}, seq_len={seq_len}, device={device}, dtype={dtype}")
            
            # Initialize memory with the correct dtype and device
            self._initialize_memory(batch_size, device, dtype=dtype)
            
            # Process each sequence step to update memory
            for t in range(seq_len):
                # Current hidden state [batch_size, input_dim]
                current_state = hidden_states[:, t]
                
                # Transform to memory space
                memory_input = self.input_transform(current_state)  # [batch_size, memory_dim]
                self.log_debug(f"Step {t}: memory_input shape after transformation: {memory_input.shape}")
                
                # Calculate similarity with existing memory
                similarity = torch.bmm(
                    memory_input.unsqueeze(1),  # [batch_size, 1, memory_dim]
                    self.memory.transpose(1, 2)  # [batch_size, memory_dim, num_slots]
                ).squeeze(1)  # [batch_size, num_slots]
                
                self.log_debug(f"Step {t}: similarity shape: {similarity.shape}")
                
                # Rest of the memory update logic...
                # [implementation continues as before with occasional debug logging]
                
                # Log a sample of the updated memory state
                if t == seq_len - 1:  # Only on last step to avoid excessive logging
                    self.log_debug(f"Updated memory shape: {self.memory.shape}")
                    self.log_debug(f"Updated memory sample (first entry): {self.memory[0, 0, :5].tolist()}")
            
            return self.memory
            
        except Exception as e:
            self.log_debug(f"ERROR in memory controller: {str(e)}")
            self.log_debug(f"ERROR DETAILS - hidden_states shape: {hidden_states.shape}")
            self.log_debug(f"ERROR DETAILS - memory shape: {None if self.memory is None else self.memory.shape}")
            
            # Re-raise for caller handling
            raise


class MemoryAttentionGate(nn.Module):
    """Attention-based memory gate that controls information flow to/from memory"""
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_heads: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Ensure head_dim is an integer
        self.head_dim = memory_dim // num_heads
        assert memory_dim % num_heads == 0, f"Memory dimension {memory_dim} must be divisible by number of heads {num_heads}"
        
        # Multi-head attention components with matching dimensions
        self.query = nn.Linear(input_dim, memory_dim)
        self.key = nn.Linear(memory_dim, memory_dim)
        self.value = nn.Linear(memory_dim, memory_dim)
        self.output = nn.Linear(memory_dim, input_dim)
        
        # Normalization and dropout
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feedforward network
        self.ff_network = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Debugging stats
        self.debug_layer_id = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def set_debug_id(self, layer_id):
        """Set a layer identifier for debug messages"""
        self.debug_layer_id = layer_id
    
    def log_debug(self, message):
        """Log debug message with layer identifier"""
        prefix = f"[Memory Gate {self.debug_layer_id}] " if self.debug_layer_id is not None else "[Memory Gate] "
        logger.info(f"{prefix}{message}")
    
    def forward(self, x, memory):
        """
        Process input through memory attention mechanism with detailed tensor shape logging
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            memory: Memory tensor [batch_size, num_slots, memory_dim]
            
        Returns:
            output: Memory-enhanced representation [batch_size, seq_len, input_dim]
        """
        try:
            # Log input shapes
            self.log_debug(f"Input x shape: {x.shape}, Memory shape: {memory.shape}")
            self.log_debug(f"Expected dimensions - input_dim: {self.input_dim}, memory_dim: {self.memory_dim}")
            
            batch_size, seq_len, _ = x.shape
            _, num_slots, _ = memory.shape
            
            self.log_debug(f"Extracted dimensions - batch_size: {batch_size}, seq_len: {seq_len}, num_slots: {num_slots}")
            
            # Compute residual attention
            residual = x
            x_norm = self.layer_norm1(x)
            self.log_debug(f"After layer_norm1 shape: {x_norm.shape}")
            
            # Project to query, key, value spaces
            q = self.query(x_norm)
            k = self.key(memory)
            v = self.value(memory)
            
            self.log_debug(f"After projection - Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")
            
            # Start multi-head attention implementation
            # First attempt: Standard reshape for multi-head attention
            try:
                self.log_debug("Attempting standard multi-head reshaping...")
                
                # Reshape for multi-head attention
                q_reshaped = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
                self.log_debug(f"q_reshaped shape: {q_reshaped.shape}")
                
                k_reshaped = k.view(batch_size, num_slots, self.num_heads, self.head_dim)
                self.log_debug(f"k_reshaped shape: {k_reshaped.shape}")
                
                v_reshaped = v.view(batch_size, num_slots, self.num_heads, self.head_dim)
                self.log_debug(f"v_reshaped shape: {v_reshaped.shape}")
                
                # Transpose for attention computation
                q_transposed = q_reshaped.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
                self.log_debug(f"q_transposed shape: {q_transposed.shape}")
                
                k_transposed = k_reshaped.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_slots, head_dim]
                self.log_debug(f"k_transposed shape: {k_transposed.shape}")
                
                v_transposed = v_reshaped.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_slots, head_dim]
                self.log_debug(f"v_transposed shape: {v_transposed.shape}")
                
                # Scaled dot-product attention
                # [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, head_dim, num_slots]
                self.log_debug(f"k_transposed.transpose(-2, -1) shape: {k_transposed.transpose(-2, -1).shape}")
                
                attention_scores = torch.matmul(q_transposed, k_transposed.transpose(-2, -1))
                attention_scores = attention_scores / math.sqrt(self.head_dim)
                self.log_debug(f"attention_scores shape: {attention_scores.shape}")
                
                # Apply softmax to get attention weights
                attention_weights = F.softmax(attention_scores, dim=-1)
                attention_weights = self.dropout(attention_weights)
                self.log_debug(f"attention_weights shape: {attention_weights.shape}")
                
                # Apply attention weights to values
                # [batch_size, num_heads, seq_len, num_slots] x [batch_size, num_heads, num_slots, head_dim]
                context_layer = torch.matmul(attention_weights, v_transposed)
                self.log_debug(f"context_layer shape after attention: {context_layer.shape}")
                
                # Reshape back to original dimensions
                # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                self.log_debug(f"context_layer shape after permute: {context_layer.shape}")
                
                # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, memory_dim]
                context_layer = context_layer.view(batch_size, seq_len, self.memory_dim)
                self.log_debug(f"context_layer shape after view: {context_layer.shape}")
                
                # Success path - standard multi-head attention worked
                self.log_debug("Standard multi-head attention successful!")
                
            except RuntimeError as e:
                # If standard approach fails, we'll log the error and try an alternative approach
                self.log_debug(f"Standard approach failed with error: {str(e)}")
                self.log_debug("Attempting alternative attention mechanism...")
                
                # Alternative approach: batch matrix multiplication with explicit reshaping
                # Flatten sequence and batch dimensions to use batch matrix multiplication (bmm)
                q_flat = q.view(batch_size * seq_len, 1, self.memory_dim)
                self.log_debug(f"q_flat shape: {q_flat.shape}")
                
                # Expand memory for each sequence token
                k_expanded = k.unsqueeze(1).expand(batch_size, seq_len, num_slots, self.memory_dim)
                k_flat = k_expanded.reshape(batch_size * seq_len, num_slots, self.memory_dim)
                self.log_debug(f"k_expanded shape: {k_expanded.shape}, k_flat shape: {k_flat.shape}")
                
                v_expanded = v.unsqueeze(1).expand(batch_size, seq_len, num_slots, self.memory_dim)
                v_flat = v_expanded.reshape(batch_size * seq_len, num_slots, self.memory_dim)
                self.log_debug(f"v_expanded shape: {v_expanded.shape}, v_flat shape: {v_flat.shape}")
                
                # Compute attention with batch matrix multiplication
                attention_scores = torch.bmm(q_flat, k_flat.transpose(1, 2))
                attention_scores = attention_scores / math.sqrt(self.memory_dim)
                self.log_debug(f"bmm attention_scores shape: {attention_scores.shape}")
                
                attention_weights = F.softmax(attention_scores, dim=2)
                attention_weights = self.dropout(attention_weights)
                self.log_debug(f"bmm attention_weights shape: {attention_weights.shape}")
                
                context_layer = torch.bmm(attention_weights, v_flat)
                self.log_debug(f"bmm context_layer shape: {context_layer.shape}")
                
                # Reshape back to original sequence dimensions
                context_layer = context_layer.view(batch_size, seq_len, self.memory_dim)
                self.log_debug(f"Final context_layer shape: {context_layer.shape}")
                
                # Success path for alternative approach
                self.log_debug("Alternative attention mechanism successful!")
            
            # Output projection and dropout
            attn_output = self.output(context_layer)
            attn_output = self.dropout(attn_output)
            self.log_debug(f"After output projection shape: {attn_output.shape}")
            
            # First residual connection
            x_combined = residual + attn_output
            self.log_debug(f"After first residual connection shape: {x_combined.shape}")
            
            # Feedforward network with residual connection and layer norm
            residual2 = x_combined
            x_norm2 = self.layer_norm2(x_combined)
            self.log_debug(f"After layer_norm2 shape: {x_norm2.shape}")
            
            ff_output = self.ff_network(x_norm2)
            self.log_debug(f"After feedforward network shape: {ff_output.shape}")
            
            final_output = residual2 + ff_output
            self.log_debug(f"Final output shape: {final_output.shape}")
            
            return final_output
            
        except Exception as e:
            # Comprehensive error reporting
            self.log_debug(f"ERROR: {str(e)}")
            self.log_debug(f"ERROR DETAILS - input shapes: x={x.shape}, memory={memory.shape}")
            self.log_debug(f"ERROR DETAILS - model config: input_dim={self.input_dim}, memory_dim={self.memory_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}")
            
            # Re-raise to allow the calling code to handle it
            raise


class EnhancedMemoryLayer(nn.Module):
    """Memory layer that processes inputs through memory bank and attention mechanisms"""
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_slots: int = 1024,
        num_heads: int = 16,
        update_rate: float = 0.5,
        feedback_strength: float = 0.8,
        use_gru: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.feedback_strength = feedback_strength
        self.debug_layer_id = None
        
        logger.info(f"EnhancedMemoryLayer initialized with input_dim={input_dim}, memory_dim={memory_dim}, num_slots={num_slots}")
        
        # Make sure memory_dim is divisible by num_heads
        if memory_dim % num_heads != 0:
            orig_heads = num_heads
            num_heads = math.gcd(memory_dim, num_heads)
            if num_heads == 1:
                # Find largest factor less than 32
                for i in range(32, 0, -1):
                    if memory_dim % i == 0:
                        num_heads = i
                        break
            logger.warning(f"Adjusted num_heads from {orig_heads} to {num_heads} to be divisible by memory_dim {memory_dim}")
        
        # Memory controller for storage
        self.memory_controller = MemoryController(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_slots=num_slots,
            update_rate=update_rate,
            use_gru=use_gru
        )
        
        # Memory attention gate for retrieval and integration
        self.memory_gate = MemoryAttentionGate(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Input salience network
        self.salience_network = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid()
        )
        
        # Input gate
        self.input_gate = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid()
        )
        
        # Forget gate
        self.forget_gate = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid()
        )
        
        # Output gate
        self.output_gate = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
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
    
    def set_debug_id(self, layer_id):
        """Set debug ID for this layer and propagate to components"""
        self.debug_layer_id = layer_id
        self.memory_controller.set_debug_id(layer_id)
        self.memory_gate.set_debug_id(layer_id)
    
    def log_debug(self, message):
        """Log debug message with layer identifier"""
        prefix = f"[Memory Layer {self.debug_layer_id}] " if self.debug_layer_id is not None else "[Memory Layer] "
        logger.info(f"{prefix}{message}")
    
    def reset_memory(self):
        """Reset memory state"""
        self.memory_controller.reset_memory()
    
    def forward(self, hidden_states):
        """Process hidden states through memory layer with debug instrumentation"""
        try:
            self.log_debug(f"Forward called with hidden_states shape: {hidden_states.shape}")
            
            # Ensure hidden_states is a tensor, not a tuple
            if isinstance(hidden_states, tuple):
                self.log_debug(f"hidden_states is a tuple of length {len(hidden_states)}")
                hidden_states = hidden_states[0]
                self.log_debug(f"Using first element with shape: {hidden_states.shape}")
            
            # Store original for residual connection
            original_hidden = hidden_states
            
            # Calculate salience of input (for monitoring only)
            salience = self.salience_network(hidden_states)
            self.log_debug(f"Salience shape: {salience.shape}")
            
            # Input gate - decide what to write to memory
            gin = self.input_gate(hidden_states)
            self.log_debug(f"Input gate shape: {gin.shape}")
            
            # Update memory with current hidden states
            memory_input = gin * hidden_states
            self.log_debug(f"Memory input shape: {memory_input.shape}")
            
            memory = self.memory_controller(memory_input)
            self.log_debug(f"Memory shape after controller: {memory.shape}")
            
            # Forget gate - decide what to forget from memory
            gforget = self.forget_gate(memory)
            self.log_debug(f"Forget gate shape: {gforget.shape}")
            
            memory = gforget * memory
            
            # Retrieve and integrate memory with hidden states
            self.log_debug(f"Calling memory gate with shapes - hidden: {hidden_states.shape}, memory: {memory.shape}")
            enhanced_states = self.memory_gate(hidden_states, memory)
            self.log_debug(f"Enhanced states shape after memory gate: {enhanced_states.shape}")
            
            # Output gate - control memory influence
            gout = self.output_gate(memory)  # Shape [1, 4096, 1]
            
            # Create an attention mechanism to match memory dimensions to sequence dimensions
            batch_size, seq_len, hidden_dim = enhanced_states.shape
            _, num_slots, _ = memory.shape
            
            # Compute attention between enhanced states and memory
            attn_query = enhanced_states.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
            attn_scores = torch.bmm(attn_query, memory.transpose(1, 2))  # [batch_size, 1, num_slots]
            attn_weights = F.softmax(attn_scores, dim=2)  # [batch_size, 1, num_slots]
            
            # Weight the output gate by attention and sum
            weighted_gate = torch.bmm(attn_weights, gout)  # [batch_size, 1, 1]
            gate_expanded = weighted_gate.expand(-1, seq_len, -1)  # [batch_size, seq_len, 1]
            
            # Apply weighted gate to enhanced states
            memory_contribution = gate_expanded * enhanced_states
            
            # Apply memory feedback for improved integration
            output = hidden_states + self.feedback_strength * (memory_contribution - hidden_states)
            
            return output
            
        except Exception as e:
            self.log_debug(f"ERROR in memory layer: {str(e)}")
            self.log_debug(f"ERROR DETAILS - hidden_states shape: {hidden_states.shape if not isinstance(hidden_states, tuple) else 'tuple'}")
            
            # Re-raise for caller handling
            raise

#-------------------------------------------------------------------------------
# Gemma Model with Memory
#-------------------------------------------------------------------------------

class GemmaWithMemory(nn.Module):
    """Gemma model enhanced with LM2 memory mechanisms"""
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Set up device
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize hidden_size with a default value in case model config extraction fails
        self.hidden_size = 8192  # Default for Gemma-3-27B
        
        # Load model and tokenizer with HF authentication
        logger.info(f"Loading base model: {config.base_model_name}")
        
        try:
            # Hugging Face authentication if token is provided
            if config.hf_token:
                # Login to Hugging Face 
                try:
                    login(token=config.hf_token)
                    logger.info("Successfully authenticated with Hugging Face")
                except Exception as e:
                    logger.error(f"Failed to authenticate with Hugging Face: {e}")
                    raise
            
            # Load tokenizer with token
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name,
                token=config.hf_token
            )
            
            # Prepare model loading kwargs
            model_kwargs = {
                "device_map": "cpu" if not config.offload_to_cpu else "cpu",
                "token": config.hf_token if config.hf_token else None,
                "torch_dtype": torch.float16 if config.use_half_precision else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            # Add quantization if requested
            if config.use_4bit_quantization:
                model_kwargs.update({
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.float16,
                })
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    logger.warning("BitsAndBytes not available, using standard 16-bit loading")
            
            # Enable gradient checkpointing if requested
            if config.gradient_checkpointing:
                model_kwargs["use_gradient_checkpointing"] = True
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name, 
                **model_kwargs
            )
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
                
            logger.info(f"Successfully loaded model: {config.base_model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Debug: print model structure
        logger.info("Model structure debugging:")
        for name, _ in self.model.named_children():
            logger.info(f"- {name}")
            if hasattr(self.model, name):
                submodule = getattr(self.model, name)
                for subname, _ in submodule.named_children():
                    logger.info(f"  - {subname}")
        
        # CRITICAL: Extract hidden size from model config BEFORE creating memory layers
        self._extract_hidden_size()
        logger.info(f"Model hidden size: {self.hidden_size}")
        
        # After loading the model, print detected dimensions
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'model_dim'):
            self.hidden_size = self.model.config.model_dim
        else:
            self.hidden_size = 5376  # Use the dimension from the error message
        
        logger.info(f"Model hidden size: {self.hidden_size}")
        logger.info(f"Memory dimension: {config.memory_dim}")
        
        # Ensure memory_dim matches hidden_size
        if config.memory_dim != self.hidden_size:
            logger.warning(f"Memory dimension ({config.memory_dim}) doesn't match model hidden size ({self.hidden_size})")
            logger.warning(f"Setting memory_dim = {self.hidden_size}")
            config.memory_dim = self.hidden_size
        
        # Now create memory layers with the correct dimension

        # Create memory layers - AFTER hidden_size is properly set
        logger.info(f"Creating {config.num_memory_layers} memory enhancement layers")
        self.memory_layers = nn.ModuleList([
            EnhancedMemoryLayer(
                input_dim=self.hidden_size,
                memory_dim=config.memory_dim,
                num_slots=config.num_memory_slots,
                num_heads=config.attention_heads,
                update_rate=config.memory_update_rate,
                feedback_strength=config.feedback_strength,
                use_gru=config.use_gru_controller,
                dropout=config.dropout_rate,
                #use_dimension_reduction=False,  # Enable dimension reduction for CPU usage
                #projection_factor=0.25  # Reduce memory footprint
            ) for _ in range(config.num_memory_layers)
        ])
        
        # Register hooks to intercept hidden states
        self.hidden_states_dict = {}
        
        # Find the layers in the model
        self.model_layers = self._find_model_layers()
        
        # Layer merging maps for matching memory layers to model layers
        num_model_layers = len(self.model_layers)
        if config.num_memory_layers > num_model_layers:
            logger.warning(f"Requested {config.num_memory_layers} memory layers but model only has {num_model_layers} layers")
            config.num_memory_layers = num_model_layers
        
        # Create mapping (evenly distribute memory layers across transformer layers)
        if config.num_memory_layers == 2:
            # Just use first and middle layers
            self.layer_mapping = {0: 0, 61: 1}
        if config.num_memory_layers == 10:
            # Just use first and middle layers
            self.layer_mapping = {5: 0, 11: 1, 17: 2, 23: 3, 29: 4, 35: 5, 41: 6, 47: 7, 53: 8, 59: 9}

        self.layer_mapping = {}
        if config.num_memory_layers == num_model_layers:
            # One-to-one mapping
            self.layer_mapping = {i: i for i in range(num_model_layers)}

        if config.num_memory_layers == 2:
            # Just use first and middle layers
            self.layer_mapping = {0: 0, 61: 1}
        if config.num_memory_layers == 10:
            # Just use first and middle layers
            self.layer_mapping = {5: 0, 11: 1, 17: 2, 23: 3, 29: 4, 35: 5, 41: 6, 47: 7, 53: 8, 59: 9}
        else:
            # Distribute evenly
            indices = torch.linspace(0, num_model_layers-1, config.num_memory_layers).long().tolist()
            self.layer_mapping = {i: idx for idx, i in enumerate(indices)}
        
        logger.info(f"Created layer mapping: {self.layer_mapping}")
        
        # Register hooks to intercept hidden states
        self._register_hooks()
        
        # Store blend ratio for memory integration
        self.blend_ratio = config.blend_ratio
        
        logger.info("Memory-enhanced Gemma model initialized successfully")
    
    def _extract_hidden_size(self):
        """Extract hidden size from model config with multiple fallbacks"""
        # Try multiple ways to get the hidden size
        try:
            # First try the config attribute for hidden_size
            if hasattr(self.model.config, 'hidden_size'):
                self.hidden_size = self.model.config.hidden_size
                logger.info(f"Found hidden_size in model.config.hidden_size: {self.hidden_size}")
                return
                
            # Next try model_dim (used by Gemma-3)
            elif hasattr(self.model.config, 'model_dim'):
                self.hidden_size = self.model.config.model_dim
                logger.info(f"Found hidden_size in model.config.model_dim: {self.hidden_size}")
                return
                
            # Next try d_model (used by some transformer architectures)
            elif hasattr(self.model.config, 'd_model'):
                self.hidden_size = self.model.config.d_model
                logger.info(f"Found hidden_size in model.config.d_model: {self.hidden_size}")
                return
                
            # Check for Gemma & LLaMA specific parameters
            elif hasattr(self.model.config, 'hidden_dim'):
                self.hidden_size = self.model.config.hidden_dim
                logger.info(f"Found hidden_size in model.config.hidden_dim: {self.hidden_size}")
                return
                
            # Try to extract from model structure for certain architectures
            for key, value in vars(self.model.config).items():
                if any(size_key in key.lower() for size_key in ['hidden', 'dim', 'size', 'width']):
                    if isinstance(value, int) and value > 256:  # Reasonable size for a hidden dimension
                        self.hidden_size = value
                        logger.info(f"Found hidden_size in model.config.{key}: {self.hidden_size}")
                        return
            
            # If still not found, attempt to inspect the model's first layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers') and len(self.model.model.layers) > 0:
                first_layer = self.model.model.layers[0]
                # Look for common layer attributes that might contain dimension info
                for name, module in first_layer.named_modules():
                    if isinstance(module, nn.Linear):
                        if module.in_features > 256:  # Reasonable size
                            self.hidden_size = module.in_features
                            logger.info(f"Inferred hidden_size from layer dimensions: {self.hidden_size}")
                            return
            
            # If we reach here, use the default
            logger.warning(f"Could not determine hidden size from model. Using default: {self.hidden_size}")
            
        except Exception as e:
            logger.error(f"Error extracting hidden size: {e}. Using default: {self.hidden_size}")
    
    # [The rest of the methods remain the same]
    
    def _find_model_layers(self):
        """Find the transformer layers in the model architecture"""
        # Common patterns for transformer layers in different architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            logger.info("Found layers at model.model.layers")
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            logger.info("Found layers at model.transformer.layers")
            return self.model.transformer.layers
        elif hasattr(self.model, 'layers'):
            logger.info("Found layers at model.layers")
            return self.model.layers
        
        # For Gemma-3 specifically, try common submodules
        for name in ['model', 'decoder', 'encoder', 'transformer']:
            if hasattr(self.model, name):
                submodule = getattr(self.model, name)
                if hasattr(submodule, 'layers'):
                    logger.info(f"Found layers at model.{name}.layers")
                    return submodule.layers
                # Check one level deeper
                for subname in ['decoder', 'encoder', 'transformer', 'model']:
                    if hasattr(submodule, subname):
                        subsubmodule = getattr(submodule, subname)
                        if hasattr(subsubmodule, 'layers'):
                            logger.info(f"Found layers at model.{name}.{subname}.layers")
                            return subsubmodule.layers
        
        # If still not found, search all model children recursively
        layers = self._search_for_layers(self.model)
        if layers is not None:
            return layers
        
        # If we get here, we couldn't find the layers
        error_msg = "Could not find layers in the model. Model structure: "
        error_msg += ", ".join([name for name, _ in self.model.named_children()])
        logger.error(error_msg)
        raise AttributeError("Could not identify layers in the model. Please check model structure.")
    
    def _search_for_layers(self, module, path=""):
        """Recursively search for a 'layers' attribute in the module hierarchy"""
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            
            # Check if this module has a 'layers' attribute that looks like transformer layers
            if hasattr(child, 'layers') and isinstance(child.layers, (list, nn.ModuleList)):
                layers = child.layers
                # Verify these look like transformer layers (they should have attention)
                if len(layers) > 0:
                    first_layer = layers[0]
                    for subname, _ in first_layer.named_children():
                        if 'attention' in subname.lower():
                            logger.info(f"Found layers at {current_path}.layers")
                            return layers
            
            # Recursively search children
            result = self._search_for_layers(child, current_path)
            if result is not None:
                return result
        
        return None
    
    def _register_hooks(self):
        """Register hooks for hidden state manipulation with debug IDs"""
        def get_hook_fn(layer_idx):
            """Create a hook function for a specific layer"""
            def hook_fn(module, input_tensors, output_tensors):
                # Handle different output formats (some models return tuples)
                is_tuple = isinstance(output_tensors, tuple)
                
                if is_tuple:
                    # Extract the hidden states and remember secondary outputs
                    hidden_states = output_tensors[0]
                    extra_outputs = output_tensors[1:]
                else:
                    hidden_states = output_tensors
                    extra_outputs = None
                
                # Store the hidden states for reference
                self.hidden_states_dict[layer_idx] = hidden_states
                
                # Process through memory layer if this layer has a mapping
                if layer_idx in self.layer_mapping:
                    memory_idx = self.layer_mapping[layer_idx]
                    
                    try:
                        logger.info(f"Processing layer {layer_idx} with memory layer {memory_idx}, hidden_states shape: {hidden_states.shape}")
                        
                        # Process through memory layer
                        enhanced = self.memory_layers[memory_idx](hidden_states)
                        
                        logger.info(f"Layer {layer_idx} processing complete, enhanced shape: {enhanced.shape}")
                        
                        # Return in the original format
                        if is_tuple:
                            return (enhanced,) + extra_outputs
                        else:
                            return enhanced
                    except Exception as e:
                        logger.error(f"Error in memory layer {memory_idx} for model layer {layer_idx}: {e}")
                        # Fall back to original if memory processing fails
                        return output_tensors
                else:
                    # No memory processing for this layer
                    return output_tensors
            
            return hook_fn
        
        # Set debug IDs for memory layers
        # FIX: Correctly iterate through the mapping (model_layer → memory_layer)
        for model_idx, memory_idx in self.layer_mapping.items():
            # Now properly accessing memory layers by their index
            self.memory_layers[memory_idx].set_debug_id(f"{memory_idx}-{model_idx}")
        
        # Register hooks for each transformer layer
        for i, layer in enumerate(self.model_layers):
            layer.register_forward_hook(get_hook_fn(i))
        
        logger.info(f"Registered hooks for {len(self.model_layers)} layers")
    
    def reset_memory(self):
        """Reset all memory layers"""
        for layer in self.memory_layers:
            layer.reset_memory()
        logger.info("Memory state reset")
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass with memory augmentation"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Hidden states are already processed by hooks
        return outputs
    
    def generate(self, 
                 input_ids=None, 
                 attention_mask=None, 
                 max_length=None, 
                 min_length=None,
                 do_sample=True,
                 temperature=0.8,
                 top_p=0.9,
                 top_k=50,
                 repetition_penalty=1.1,
                 **kwargs):
        """Generation with memory enhancement"""
        generation_config = {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            **kwargs
        }
        
        # Create stopping criteria for </assistant> token
        stopping_criteria = kwargs.get("stopping_criteria", StoppingCriteriaList())
        stopping_criteria.append(
            AssistantEndTokenCriteria(self.tokenizer, self.device)
        )
        
        # Add to generation config
        kwargs["stopping_criteria"] = stopping_criteria
        

        
        # Filter out None values
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        
        # Memory is updated through the forward hooks during generation
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
    
    def get_memory_state(self):
        """Extract current memory states for analysis/visualization"""
        memory_states = []
        for i, layer in enumerate(self.memory_layers):
            if hasattr(layer.memory_controller, 'memory') and layer.memory_controller.memory is not None:
                memory_states.append({
                    'layer': i,
                    'memory': layer.memory_controller.memory.detach().cpu().numpy(),
                    'usage': layer.memory_controller.usage.detach().cpu().numpy(),
                    'age': layer.memory_controller.age.detach().cpu().numpy()
                })
        return memory_states
    
    def save_memory_state(self, path):
        """Save memory state to disk"""
        memory_states = {}
        for i, layer in enumerate(self.memory_layers):
            if hasattr(layer.memory_controller, 'memory') and layer.memory_controller.memory is not None:
                memory_states[f'layer_{i}_memory'] = layer.memory_controller.memory.detach().cpu()
                memory_states[f'layer_{i}_usage'] = layer.memory_controller.usage.detach().cpu()
                memory_states[f'layer_{i}_age'] = layer.memory_controller.age.detach().cpu()
        
        torch.save(memory_states, path)
        logger.info(f"Memory state saved to {path}")
        
    def load_memory_state(self, path):
        """Load memory state from disk"""
        #if not torch.cuda.is_available():
        memory_states = torch.load(path, map_location=torch.device('cpu'))
        #else:
           #memory_states = torch.load(path)
        
        for i, layer in enumerate(self.memory_layers):
            if f'layer_{i}_memory' in memory_states:
                layer.memory_controller.memory = memory_states[f'layer_{i}_memory'].to(self.device)
                layer.memory_controller.usage = memory_states[f'layer_{i}_usage'].to(self.device)
                layer.memory_controller.age = memory_states[f'layer_{i}_age'].to(self.device)
        
        logger.info(f"Memory state loaded from {path}")

#-------------------------------------------------------------------------------
# Chat Interface
#-------------------------------------------------------------------------------

def format_conversation(conversation: List[Dict[str, str]]) -> str:
    """Format conversation for model input"""
    formatted = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            formatted += f"<user>\n{content}\n</user>\n"
        elif role == "assistant":
            formatted += f"<assistant>\n{content}\n</assistant>\n"
        elif role == "system":
            formatted += f"<system>\n{content}\n</system>\n"
    
    # Add final assistant prompt
    formatted += "<assistant>\n"
    return formatted


class ChatInterface:
    """Chat interface for interacting with memory-enhanced model"""
    def __init__(
        self,
        model_config: MemoryConfig,
        conversation_memory_path: Optional[str] = None,
        model_memory_path: Optional[str] = None,
        system_prompt: str = ""
    ):
        self.config = model_config
        self.conversation_memory_path = conversation_memory_path
        self.model_memory_path = model_memory_path
        
        # Initialize model
        self.model = GemmaWithMemory(model_config)
        
        # Load model memory if exists
        if model_memory_path and os.path.exists(model_memory_path):
            self.model.load_memory_state(model_memory_path)
        
        # Initialize conversation history
        self.conversation = []
        if system_prompt:
            self.conversation.append({"role": "system", "content": system_prompt})
        
        # Load conversation if exists
        if conversation_memory_path and os.path.exists(conversation_memory_path):
            try:
                with open(conversation_memory_path, 'r') as f:
                    self.conversation = json.load(f)
                logger.info(f"Loaded conversation from {conversation_memory_path}")
            except:
                logger.warning(f"Failed to load conversation from {conversation_memory_path}, starting fresh")
    
    def save_state(self):
        """Save conversation and model memory state"""
        if self.conversation_memory_path:
            with open(self.conversation_memory_path, 'w') as f:
                json.dump(self.conversation, f, indent=2)
            logger.info(f"Saved conversation to {self.conversation_memory_path}")
        
        if self.model_memory_path:
            self.model.save_memory_state(self.model_memory_path)
    
    def add_user_message(self, message: str):
        """Add user message to conversation"""
        self.conversation.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        """Add assistant message to conversation"""
        self.conversation.append({"role": "assistant", "content": message})
    
    def get_assistant_response(
        self,
        temperature: float = 0.8,
        max_new_tokens: int = 8192,
        top_p: float = 0.95
    ) -> str:
        """Generate response from assistant"""
        # Format conversation
        prompt = format_conversation(self.conversation)
        
        # Tokenize
        inputs = self.model.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        # Generate response
        try:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.model.tokenizer.eos_token_id
            )
            
            # Extract only the newly generated tokens
            new_tokens = output_ids[0, input_ids.shape[1]:]
            response = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Add to conversation history
            self.add_assistant_message(response)
            
            # Save state
            self.save_state()
            
            return response
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            error_response = f"I'm sorry, I encountered an error: {str(e)}"
            self.add_assistant_message(error_response)
            return error_response
    
    def chat_turn(self, user_message: str) -> str:
        """Process one turn of conversation"""
        self.add_user_message(user_message)
        return self.get_assistant_response()
    
    def start_interactive_session(self):
        """Start an interactive chat session using a file-based approach"""
        import os
        import time
        import tempfile
        import signal
        import sys
        
        # Create a temporary file for input
        temp_dir = tempfile.gettempdir()
        input_file = os.path.join(temp_dir, "gemma_memory_input.txt")
        input_ready_file = os.path.join(temp_dir, "gemma_memory_ready.txt")
        
        # Setup signal handler for clean exit
        def signal_handler(sig, frame):
            print(f"\n{Fore.YELLOW}Interrupt received. Cleaning up...{Style.RESET_ALL}")
            # Clean up input files
            for file_path in [input_file, input_ready_file]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            # Save state before exiting
            self.save_state()
            print(f"{Fore.GREEN}Chat session ended. States saved.{Style.RESET_ALL}")
            sys.exit(0)
            
        # Register signal handler for clean termination
        signal.signal(signal.SIGINT, signal_handler)
        
        print(f"{Fore.GREEN}Starting interactive chat session.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Due to console input issues, we'll use a file-based approach:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}1. To send a message, write it to: {input_file}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}2. Then create an empty file at: {input_ready_file}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}3. Type 'exit' to end, 'reset memory' to clear model memory.{Style.RESET_ALL}")
        
        # Clear any existing input files
        for file_path in [input_file, input_ready_file]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove file {file_path}: {e}")
        
        # Create empty input file
        try:
            with open(input_file, 'w') as f:
                f.write("")
        except Exception as e:
            print(f"Error creating input file: {e}")
            return
        
        # Print system prompt if exists
        if self.conversation and self.conversation[0]["role"] == "system":
            print(f"{Fore.CYAN}System: {self.conversation[0]['content']}{Style.RESET_ALL}")
        
        # Print any existing conversation
        for turn in self.conversation:
            if turn["role"] == "system":
                continue
            if turn["role"] == "user":
                print(f"{Fore.BLUE}User: {turn['content']}{Style.RESET_ALL}")
            elif turn["role"] == "assistant":
                print(f"{Fore.GREEN}Assistant: {turn['content']}{Style.RESET_ALL}")
        
        # Interactive loop
        try:
            while True:
                print(f"{Fore.BLUE}Waiting for input... (Ctrl+C to exit){Style.RESET_ALL}")
                
                # Wait for ready file to appear
                while not os.path.exists(input_ready_file):
                    time.sleep(0.5)
                
                # Read input from file
                try:
                    with open(input_file, 'r') as f:
                        user_input = f.read().strip()
                    
                    # Remove the ready file
                    if os.path.exists(input_ready_file):
                        os.remove(input_ready_file)
                    
                    print(f"{Fore.BLUE}User: {user_input}{Style.RESET_ALL}")
                    
                    if user_input.lower() == 'exit':
                        break
                    
                    if user_input.lower() == 'reset memory':
                        self.model.reset_memory()
                        print(f"{Fore.YELLOW}Memory has been reset.{Style.RESET_ALL}")
                        continue
                    
                    # Get response
                    print(f"{Fore.GREEN}Assistant: {Style.RESET_ALL}", end="", flush=True)
                    
                    # Process message and stream output
                    self.add_user_message(user_input)
                    response = self.get_assistant_response()
                    print(response)
                    
                except Exception as e:
                    print(f"{Fore.RED}Error processing input: {e}{Style.RESET_ALL}")
                    
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Keyboard interrupt detected. Ending session.{Style.RESET_ALL}")
        finally:
            # Clean up input files
            for file_path in [input_file, input_ready_file]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    
            # Save final state
            self.save_state()
            print(f"{Fore.GREEN}Chat session ended. States saved.{Style.RESET_ALL}")
#-------------------------------------------------------------------------------
# Main Entry Point
#-------------------------------------------------------------------------------

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LM2 Memory-Enhanced Gemma Chat Interface")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        help="Base model to use")
    parser.add_argument("--memory_dim", type=int, default=768,
                        help="Memory dimension")
    parser.add_argument("--num_slots", type=int, default=1024,
                        help="Number of memory slots")
    parser.add_argument("--memory_layers", type=int, default=10,
                        help="Number of memory layers")
    parser.add_argument("--feedback", type=float, default=0.8,
                        help="Memory feedback strength")
    parser.add_argument("--blend", type=float, default=0.7,
                        help="Memory blend ratio")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision (float16)", default=False)
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization", default=False)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API token for authentication")
    parser.add_argument("--hf_token_file", type=str, default=None,
                        help="Path to file containing Hugging Face API token")
    
    # Memory persistence
    parser.add_argument("--memory_dir", type=str, default="./memory",
                        help="Directory to save/load memory state")
    parser.add_argument("--reset", action="store_true",
                        help="Reset memory state")
    
    # Chat options
    parser.add_argument("--system_prompt", type=str, 
                        default="You are an assistant with excellent memory capabilities.",
                        help="System prompt")
    
    args = parser.parse_args()
    
    # Handle Hugging Face token
    hf_token = args.hf_token
    
    # Try to load from file if specified
    if args.hf_token_file and not hf_token:
        try:
            with open(args.hf_token_file, 'r') as f:
                hf_token = f.read().strip()
            logger.info(f"Loaded Hugging Face token from {args.hf_token_file}")
        except Exception as e:
            logger.warning(f"Failed to load token from file: {e}")
    
    # Prompt user for token if still not available and using a model that requires it
    if not hf_token and "google/gemma" in args.model:
        logger.info("Gemma models require Hugging Face authentication")
        hf_token = getpass.getpass("Enter your Hugging Face token: ")
    
    # Create memory directory if it doesn't exist
    os.makedirs(args.memory_dir, exist_ok=True)
    
    # Setup paths
    conversation_path = os.path.join(args.memory_dir, "conversation.json")
    model_memory_path = os.path.join(args.memory_dir, "model_memory.pt")
    
    # Check if memory should be reset
    if args.reset:
        if os.path.exists(conversation_path):
            os.remove(conversation_path)
        if os.path.exists(model_memory_path):
            os.remove(model_memory_path)
        logger.info("Memory state reset")
    
    # Configure model
    config = MemoryConfig(
        base_model_name=args.model,
        hf_token=hf_token,
        memory_dim=args.memory_dim,
        num_memory_slots=args.num_slots,
        num_memory_layers=args.memory_layers,
        feedback_strength=args.feedback,
        blend_ratio=args.blend,
        use_half_precision=args.half,
        use_4bit_quantization=args.quantize,
        offload_to_cpu=args.cpu
    )
    
    # Check for CUDA
    #if torch.cuda.is_available():
    #    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
     #   gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
      #  logger.info(f"GPU memory: {gpu_mem:.2f} GB")
        
       # if gpu_mem < 24 and not args.quantize and args.model == "google/gemma-3-27b-it":
        #    logger.warning(f"Less than 24GB VRAM detected for 27B model.")
         #   logger.warning("Enabling 4-bit quantization to reduce memory footprint")
    config.use_4bit_quantization = False  # Fixed bug: was set to False in original code
            
        #if gpu_mem < 8:
            #logger.warning("Less than 8GB VRAM detected, forcing CPU mode")
            #config.offload_to_cpu = True
    #else:
      #  logger.warning("CUDA not available, using CPU")
    config.offload_to_cpu = True
    
    # Initialize chat interface
    chat = ChatInterface(
        model_config=config,
        conversation_memory_path=conversation_path,
        model_memory_path=model_memory_path,
        system_prompt=args.system_prompt
    )
    
    # Start interactive session
    chat.start_interactive_session()

if __name__ == "__main__":
    main()