# %%
import json
import torch
from typing import List, Dict, Tuple
from collections import defaultdict
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file
from pathlib import Path

# %%
def load_graph(graph):
  with open(graph, 'r') as f:
      graph_data = json.load(f)
  return graph_data

# %%
graph_data = load_graph("attribution_graph_gemma-2-2b_gemma-fact-dallas-austin.json")
# %%
# list of supernodes
neuronpedia_supernodes = graph_data["qParams"]["supernodes"]
# %%
neuronpedia_supernodes[0][1]
# %%
def parse_supernodes_by_feature(supernodes_list: List[List]) -> Dict[int, List[int]]:
    """
    Parse Neuronpedia supernodes into layer -> features mapping
    
    Args:
        supernodes_list: List of [label, feature_1, feature_2, ...] where
                        features are formatted as 'layer_feature_tokenPos'
    
    Returns:
        Dict mapping layer_idx -> list of feature indices
    """
    supernode_features = defaultdict(list)
    
    for supernode in supernodes_list:
        label = supernode[0]  # First element is the label
        features = supernode[1:]  # Rest are feature specs
        supernode_features[label] = features
      
    

    
    return supernode_features
# %%
features_by_supernode = parse_supernodes_by_feature(neuronpedia_supernodes)

# %%
def select_supernode(features_by_supernode: Dict, label: str) -> Dict[int, List[int]]:
    """Get features for a specific concept to unlearn"""
    if label not in features_by_supernode:
        print(f"Available: {list(features_by_supernode.keys())}")
        return {}
    
    feature_specs = features_by_supernode[label]
    layer_features = defaultdict(set)
    
    for feat_spec in feature_specs:
        parts = feat_spec.split('_')
        layer_idx = int(parts[0])
        feature_idx = int(parts[1])
        layer_features[layer_idx].add(feature_idx)
    
    return {layer: sorted(list(feats)) for layer, feats in layer_features.items()}
# %%
concept_label = neuronpedia_supernodes[0][0]
concept_features = select_supernode(features_by_supernode, concept_label)
# %%
def load_transcoder_layer(transcoder_path: str, layer: int):
    """
    Load a single transcoder layer from safetensors.
    
    Args:
        transcoder_path: Path to the transcoder directory (e.g., 'gemmma-scope-2b-pt-transcoders')
        layer: Layer number to load
    
    Returns:
        dict: Dictionary with encoder and decoder weights and config
    """
    layer_path = Path(transcoder_path) / f"layers.{layer}.mlp"
    
    # Load config
    with open(layer_path / "cfg.json", 'r') as f:
        config = json.load(f)
    
    # Load SAE weights
    sae_weights = load_file(str(layer_path / "sae.safetensors"))
    
    return {
        'config': config,
        'weights': sae_weights,
        'layer': layer
    }

# %%
# Example: Load layer 0 transcoder
transcoder_layer_0 = load_transcoder_layer("gemmma-scope-2b-pt-transcoders", 0)
print(f"Loaded layer 0 transcoder")
print(f"Config: {transcoder_layer_0['config']}")
print(f"Weight keys: {list(transcoder_layer_0['weights'].keys())}")
print(f"Weight shapes: {[(k, v.shape) for k, v in transcoder_layer_0['weights'].items()]}")
# %%
def load_all_transcoders(transcoder_path: str, layers: List[int] = None):
    """
    Load transcoders for multiple layers.
    
    Args:
        transcoder_path: Path to the transcoder directory
        layers: List of layer numbers to load (None = all layers 0-25)
    
    Returns:
        dict: {layer_num: transcoder_data}
    """
    if layers is None:
        layers = range(26)  # Gemma-2-2b has 26 layers (0-25)
    
    transcoders = {}
    for layer in layers:
        try:
            transcoders[layer] = load_transcoder_layer(transcoder_path, layer)
            print(f"Loaded layer {layer}")
        except FileNotFoundError:
            print(f"Warning: Layer {layer} not found, skipping")
    
    return transcoders

# %%
# Load all transcoders
all_transcoders = load_all_transcoders("gemmma-scope-2b-pt-transcoders")
# %%
from pathlib import Path
base = Path("gemmma-scope-2b-pt-transcoders")
print("Directory structure:")
for item in sorted(base.glob("**/sae*.safetensors"))[:5]:
    print(f"  {item}")
# %%
all_transcoders[0]["weights"].keys()

# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from typing import List, Dict
from collections import defaultdict

# %% Load Model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.pad_token = tokenizer.eos_token

class GemmaScopeTranscoderCorrect:
    """
    Gemma Scope Transcoder - CORRECT implementation with dtype handling
    """
    
    def __init__(self, transcoder_data, layer_idx, model_layer=None, dtype=torch.bfloat16):
        weights = transcoder_data['weights']
        config = transcoder_data.get('config', {})
        
        # Get device from model layer (handles multi-GPU)
        if model_layer is not None:
            device = next(model_layer.parameters()).device
        else:
            device = torch.device('cuda')
        
        # Load weights with correct dtype AND device
        self.W_enc = weights['encoder.weight'].to(device=device, dtype=dtype)
        self.b_enc = weights['encoder.bias'].to(device=device, dtype=dtype)
        self.W_dec = weights['W_dec'].to(device=device, dtype=dtype)
        self.b_dec = weights['b_dec'].to(device=device, dtype=dtype)
        self.post_enc = weights['post_enc'].to(device=device, dtype=dtype)
        
        self.d_sae, self.d_in = self.W_enc.shape
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype
        
        # Config parameters
        self.k = config.get('k', 256)
        self.use_topk = config.get('activation', 'topk') == 'topk'
        self.post_encoder_scale = config.get('post_encoder_scale', False)
        self.train_post_encoder = config.get('train_post_encoder', True)
        
        print(f"Layer {layer_idx}: d_in={self.d_in}, d_sae={self.d_sae}, device={device}, dtype={dtype}")
    
    def encode(self, x):
        """
        Encode with dtype consistency
        
        Formula: features = top_k(ReLU(x @ W_enc.T + b_enc + post_enc), k=256)
        """
        # Ensure input has correct dtype (device already matches)
        x = x.to(dtype=self.dtype)
        
        # Linear transformation
        pre_activation = x @ self.W_enc.T + self.b_enc
        
        # Add post_enc (additive, not multiplicative)
        pre_activation = pre_activation + self.post_enc
        
        # ReLU
        features = F.relu(pre_activation)
        
        # Top-k
        if self.use_topk:
            features = self._apply_topk(features)
        
        return features
    
    def _apply_topk(self, features):
        """Keep only top-k features"""
        topk_values, topk_indices = torch.topk(features, k=self.k, dim=-1)
        features_topk = torch.zeros_like(features)
        features_topk.scatter_(-1, topk_indices, topk_values)
        return features_topk
    
    def decode(self, features):
        """Decode with dtype consistency"""
        features = features.to(dtype=self.dtype)
        return features @ self.W_dec + self.b_dec
    
    def forward(self, x):
        """Full reconstruction"""
        return self.decode(self.encode(x))

# %% Load all transcoders with CORRECT device mapping
wrapped_transcoders_correct = {}

for layer, tc_data in all_transcoders.items():
    wrapped_transcoders_correct[layer] = GemmaScopeTranscoderCorrect(
        tc_data, 
        layer,
        model_layer=model.model.layers[layer],  # ✅ Pass model layer, not device!
        dtype=torch.bfloat16
    )


print(f"\n✓ Loaded {len(wrapped_transcoders_correct)} transcoders")


# %% Now test reconstruction should work
# test_layer = list(wrapped_transcoders_correct.keys())[0]
# test_reconstruction_correct(
#     wrapped_transcoders_correct[test_layer], 
#     model, 
#     tokenizer, 
#     test_layer
# )
# %% Trancoder ablation
class TranscoderAblation:
    """Ablation using your locally-loaded transcoders"""
    
    def __init__(self, transcoders_dict, layer_features):
        """
        Args:
            transcoders_dict: {layer_idx: GemmaScopeTranscoder}
            layer_features: {layer_idx: [feature_indices_to_ablate]}
        """
        self.transcoders = transcoders_dict
        self.layer_features = layer_features
        
        print("Initialized ablation:")
        for layer, features in layer_features.items():
            if layer in transcoders_dict:
                print(f"  Layer {layer}: ablating {len(features)} features")
    
    def ablate(self, layer_idx, activations):
        """
        Ablate features in activations
        
        Args:
            activations: (..., d_model) - MLP outputs
        
        Returns:
            ablated: (..., d_model) - Activations with features zeroed
            delta: (..., d_model) - What was removed
        """
        if layer_idx not in self.transcoders:
            return activations, torch.zeros_like(activations)
        
        tc = self.transcoders[layer_idx]
        
        # Encode (handles top-k)
        features = tc.encode(activations)
        
        # Reconstruct original (without ablation)
        original = tc.decode(features)
        
        # Ablate selected features
        features_ablated = features.clone()
        features_ablated[..., self.layer_features[layer_idx]] = 0
        
        # Decode ablated
        ablated = tc.decode(features_ablated)
        
        # Compute delta (what was removed)
        delta = original - ablated
        
        return ablated, delta

# Create ablation handler with YOUR loaded transcoders and features
ablation = TranscoderAblation(
    wrapped_transcoders_correct,  # Your loaded transcoders
    concept_features              # Your parsed features from attribution graph
)

# %% Step 2: Test Ablation on Real Data
print("\n" + "="*60)
print("TESTING ABLATION ON REAL DATA")
print("="*60)

# Test on actual model activations
test_prompt = "The capital of Texas is Austin"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

# Collect MLP activations for testing
test_activations = {}

def collect_activations(layer_idx):
    def hook(module, input, output):
        test_activations[layer_idx] = output.detach()
    return hook

# Hook to collect activations
handles = []
for layer_idx in concept_features.keys():
    h = model.model.layers[layer_idx].mlp.register_forward_hook(
        collect_activations(layer_idx)
    )
    handles.append(h)

# Run forward pass
with torch.no_grad():
    _ = model(**inputs)

# Remove hooks
for h in handles:
    h.remove()

# Test ablation on collected activations
print(f"\nTesting ablation on: '{test_prompt}'")
for layer_idx in sorted(concept_features.keys()):
    if layer_idx in test_activations:
        acts = test_activations[layer_idx]
        ablated, delta = ablation.ablate(layer_idx, acts)
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Original shape: {acts.shape}")
        print(f"  Original norm: {acts.norm():.4f}")
        print(f"  Ablated norm: {ablated.norm():.4f}")
        print(f"  Delta norm: {delta.norm():.4f}")
        print(f"  Delta/Original: {(delta.norm() / acts.norm() * 100):.2f}%")
        print(f"  Features ablated: {len(concept_features[layer_idx])}")

# %% Step 3: Load Training Data
from datasets import load_dataset

print("\n" + "="*60)
print("LOADING TRAINING DATA")
print("="*60)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

train_texts = []
for item in dataset:
    text = item['text'].strip()
    if len(text.split()) > 20:
        train_texts.append(text)
    if len(train_texts) >= 2000:  # Use 2000 samples
        break

print(f"✓ Prepared {len(train_texts)} training texts")

# %% Step 4: Train Unlearning LoRA
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
import torch.nn.functional as F

def train_unlearning_lora(
    base_model,
    ablation_handler: TranscoderAblation,
    train_texts: List[str],
    tokenizer,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
):
    """
    Train LoRA to permanently replicate ablation
    """
    print("\n" + "="*60)
    print("TRAINING UNLEARNING LORA")
    print("="*60)
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    
    optimizer = AdamW(lora_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        import random
        texts = train_texts.copy()
        random.shuffle(texts)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to("cuda")
            
            # Get TARGET from base model
            target_outputs = {}
            
            def capture_ablated(layer_idx):
                def hook(module, input, output):
                    ablated, _ = ablation_handler.ablate(layer_idx, output)
                    # ✅ Detach AND move to cuda:0 immediately
                    target_outputs[layer_idx] = ablated.detach().to('cuda:0')
                return hook
            
            handles = []
            with torch.no_grad():
                for layer_idx in ablation_handler.layer_features.keys():
                    h = base_model.model.layers[layer_idx].mlp.register_forward_hook(
                        capture_ablated(layer_idx)
                    )
                    handles.append(h)
                
                _ = base_model(**inputs)
                
                for h in handles:
                    h.remove()
            
            # Get CURRENT from LoRA model
            lora_outputs = {}
            
            def capture_lora(layer_idx):
                def hook(module, input, output):
                    # ✅ Move to cuda:0 immediately
                    lora_outputs[layer_idx] = output.to('cuda:0')
                return hook
            
            handles = []
            for layer_idx in ablation_handler.layer_features.keys():
                h = lora_model.base_model.model.model.layers[layer_idx].mlp.register_forward_hook(
                    capture_lora(layer_idx)
                )
                handles.append(h)
            
            _ = lora_model(**inputs)
            
            for h in handles:
                h.remove()
            
            # Compute loss (all tensors already on cuda:0)
            loss = 0
            for layer_idx in ablation_handler.layer_features.keys():
                loss += F.mse_loss(lora_outputs[layer_idx], target_outputs[layer_idx])
            
            loss = loss / len(ablation_handler.layer_features)
            
            # ✅ ADD GRADIENT CLIPPING
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)  # Clip!
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if num_batches % 50 == 0:
                print(f"  Batch {num_batches}/{len(texts)//batch_size}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
    
    print("\n✓ Training complete!")
    return lora_model

# Train!
unlearned_model = train_unlearning_lora(
    model,
    ablation,
    train_texts,
    tokenizer,
    num_epochs=3,
    batch_size=4,
    lora_r=16,
    learning_rate=1e-5  # ✅ Lower learning rate
)


# %% Step 5: Test Unlearning Results
print("\n" + "="*60)
print("TESTING UNLEARNING RESULTS")
print("="*60)

test_prompts = [
    "The capital of Texas is",
    "What city is the capital of Texas?",
    "Austin is the capital of",
    "Texas state capital:",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Normal model
    with torch.no_grad():
        normal = model.generate(
            **inputs, 
            max_new_tokens=15, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Unlearned model
    with torch.no_grad():
        unlearned = unlearned_model.generate(
            **inputs, 
            max_new_tokens=15, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(f"\nPrompt: '{prompt}'")
    print(f"  Normal:    {tokenizer.decode(normal[0], skip_special_tokens=True)}")
    print(f"  Unlearned: {tokenizer.decode(unlearned[0], skip_special_tokens=True)}")

# %% Step 6: Save Unlearned Model
print("\n" + "="*60)
print("SAVING UNLEARNED MODEL")
print("="*60)

output_dir = "./gemma2-unlearned-lora"
unlearned_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✓ Saved to {output_dir}")

# %% Step 7: Verification - Test with Hooks (Optional)
print("\n" + "="*60)
print("VERIFICATION: Hook-based Ablation")
print("="*60)

# Test that hook-based ablation gives same result as LoRA
test_text = "The capital of Texas is"
inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

# Apply hooks to base model for comparison
class AblationHooks:
    def __init__(self, model, ablation_handler):
        self.model = model
        self.ablation = ablation_handler
        self.handles = []
    
    def apply_ablation(self, layer_idx):
        def hook(module, input, output):
            ablated, _ = self.ablation.ablate(layer_idx, output)
            return ablated
        return hook
    
    def register(self):
        for layer_idx in self.ablation.layer_features.keys():
            h = self.model.model.layers[layer_idx].mlp.register_forward_hook(
                self.apply_ablation(layer_idx)
            )
            self.handles.append(h)
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

# Test with hooks
hooks = AblationHooks(model, ablation)
hooks.register()

with torch.no_grad():
    hooked_output = model.generate(
        **inputs, max_new_tokens=15, do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

hooks.remove()

print(f"Test: '{test_text}'")
print(f"  Normal:        {tokenizer.decode(normal[0], skip_special_tokens=True)}")
print(f"  Hooked (temp): {tokenizer.decode(hooked_output[0], skip_special_tokens=True)}")
print(f"  LoRA (perm):   {tokenizer.decode(unlearned[0], skip_special_tokens=True)}")
print(f"\nIdeally, Hooked and LoRA outputs should be similar!")

# %% Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
✓ Loaded transcoders: {len(wrapped_transcoders_correct)} layers
✓ Features to ablate: {sum(len(f) for f in concept_features.values())} total
  └─ Across {len(concept_features)} layers

✓ Trained LoRA adapter to replicate ablation
✓ Saved unlearned model to: {output_dir}

Next steps for relearning experiment:
1. Test that model has forgotten the concept
2. Retrain on concept-specific data
3. Measure epochs until model relearns
4. Compare relearning rates across different features
""")

# %%
