"""
Optimized Modal data collection for dual-probe conscious control experiments.
Designed for maximum H100 GPU utilization and simultaneous blue/red color detection.
"""

import modal
import os
import json
import torch
import re
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

model_image = (
    modal.Image
        .debian_slim()
        .apt_install("git")
        .pip_install(
            "numpy<2.0",
            "transformer-lens==1.9", 
            "torch", 
            "sentencepiece", 
            "typeguard",
            "python-dotenv",
            "tqdm",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "ipython",
            "jupyter"
        )
)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
weights_volume = modal.Volume.from_name("weights", create_if_missing=True)
app = modal.App("conscious-control-collect")


@app.function(
    image=model_image,
    secrets=[hf_secret],
    gpu="H100",
    timeout=14400,
    volumes={"/weights": weights_volume},
    memory=80000,  # 80GB RAM for H100 optimization
    cpu=16,  # Maximum CPU cores for parallel processing
)   
def collect_dual_probe_activations(
    prompts: List[str], 
    system_prompt: str, 
    hook_points: List[str] = None,
    batch_size: int = 400,  # Increased for H100
    max_workers: int = 8,  # Parallel processing
) -> Dict[str, torch.Tensor]:
    """
    Collect activations for dual-probe experiments with H100 optimization.
    
    Args:
        prompts: List of prompts to process
        system_prompt: System prompt for conscious control instructions
        hook_points: List of hook points to collect from (default: key layers)
        batch_size: Batch size optimized for H100 memory
        max_workers: Number of parallel workers for CPU preprocessing
    
    Returns:
        Dictionary with activations, outputs, and dual color labels
    """
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import re
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    
    # Default hook points: early, mid, and late layers for comprehensive analysis
    if hook_points is None:
        hook_points = [
            "blocks.0.hook_resid_post",   # Early processing
            "blocks.7.hook_resid_post",   # Early-mid
            "blocks.15.hook_resid_post",  # Mid processing
            "blocks.23.hook_resid_post",  # Mid-late
            "blocks.31.hook_resid_post",  # Final layer
        ]
    
    model_name = "Llama-2-7b-chat-hf"
    local_path = "/weights/llama-2-7b-chat-hf"
    
    # Optimized model loading with H100 memory management
    print(f"Loading model on H100 with {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB VRAM")
    
    if os.path.exists(local_path):
        print(f"Loading cached model from {local_path}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            local_files_only=True,
            max_memory={0: "70GB"},  # H100 optimization
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            local_files_only=True,
            use_fast=True,  # Faster tokenization
        )
    else:
        print(f"Downloading model to {local_path}")
        os.makedirs(local_path, exist_ok=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ["HF_TOKEN"],
            device_map="cuda",
            torch_dtype=torch.float16,
            max_memory={0: "70GB"},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ["HF_TOKEN"],
            use_fast=True,
        )
        hf_model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load into TransformerLens with optimizations
    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        token=os.environ["HF_TOKEN"],
        device="cuda",
        dtype=torch.float16,
        fold_ln=True,
        fold_value_biases=False,
    )
    model.cuda()
    
    # Validate hook points
    valid_hooks = []
    for hook_point in hook_points:
        if hook_point not in model.mod_dict:
            max_layer = max([int(name.split('.')[1]) for name in model.mod_dict.keys() 
                           if name.startswith('blocks.') and 'hook_resid_post' in name])
            corrected_hook = f"blocks.{max_layer}.hook_resid_post"
            print(f"Hook {hook_point} not found. Using {corrected_hook} instead.")
            valid_hooks.append(corrected_hook)
        else:
            valid_hooks.append(hook_point)
    
    # Initialize storage for multi-layer activations
    layer_activations = {hook: [] for hook in valid_hooks}
    all_outputs = []
    all_blue_labels = []
    all_red_labels = []
    
    # Optimized batch processing
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    print(f"Processing {len(prompts)} prompts in {len(batches)} batches of size {batch_size}")
    
    # Parallel text processing function
    def process_output_batch(outputs_batch):
        """Process a batch of outputs for color detection in parallel."""
        results = []
        for output in outputs_batch:
            blue_mentioned = bool(re.search(r'\bblue\b', output.lower()))
            red_mentioned = bool(re.search(r'\bred\b', output.lower()))
            results.append((blue_mentioned, red_mentioned))
        return results
    
    # Activation collection hooks
    activations_cache = {}
    
    def create_activation_hook(hook_name):
        def activation_hook(activation, hook):
            # Store last token activation for each sequence
            activations_cache[hook_name] = activation[:, -1, :].detach().cpu().float()
            return activation
        return activation_hook
    
    # Process batches with progress tracking
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_idx, batch_prompts in enumerate(tqdm(batches, desc="Processing batches")):
            # Prepare batch prompts
            batch_full_prompts = [
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
                for user_prompt in batch_prompts
            ]
            
            # Tokenize batch with H100 optimization
            tokenized_batch = model.tokenizer(
                batch_full_prompts, 
                padding=True, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.cfg.device)
            
            # Clear activation cache
            activations_cache.clear()
            
            # Set up hooks for all layers
            hooks = [(hook_point, create_activation_hook(hook_point)) for hook_point in valid_hooks]
            
            # Generate with activation collection
            with model.hooks(hooks):
                with torch.no_grad():  # Memory optimization
                    batch_outputs = model.generate(
                        tokenized_batch.input_ids,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
            
            # Process outputs in parallel
            generated_texts = []
            for i, (output_ids, input_ids) in enumerate(zip(batch_outputs, tokenized_batch.input_ids)):
                input_length = len(input_ids)
                generated_ids = output_ids[input_length:]
                generated_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                generated_texts.append(generated_text)
            
            # Parallel color detection
            color_results = process_output_batch(generated_texts)
            
            # Store results
            all_outputs.extend(generated_texts)
            for blue_mentioned, red_mentioned in color_results:
                all_blue_labels.append(blue_mentioned)
                all_red_labels.append(red_mentioned)
            
            # Store activations for all layers
            for hook_point in valid_hooks:
                if hook_point in activations_cache:
                    layer_activations[hook_point].append(activations_cache[hook_point])
            
            # Memory cleanup for H100 optimization
            del tokenized_batch, batch_outputs
            torch.cuda.empty_cache()
            
            # Progress update
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} prompts...")
    
    # Consolidate multi-layer activations
    final_activations = {}
    for hook_point in valid_hooks:
        if layer_activations[hook_point]:
            final_activations[hook_point] = torch.cat(layer_activations[hook_point], dim=0)
            print(f"Layer {hook_point}: {final_activations[hook_point].shape}")
    
    # Summary statistics
    blue_count = sum(all_blue_labels)
    red_count = sum(all_red_labels)
    both_count = sum(1 for b, r in zip(all_blue_labels, all_red_labels) if b and r)
    neither_count = sum(1 for b, r in zip(all_blue_labels, all_red_labels) if not b and not r)
    
    print(f"\nColor Detection Summary:")
    print(f"  Blue mentions: {blue_count}/{len(all_blue_labels)} ({blue_count/len(all_blue_labels)*100:.1f}%)")
    print(f"  Red mentions: {red_count}/{len(all_red_labels)} ({red_count/len(all_red_labels)*100:.1f}%)")
    print(f"  Both colors: {both_count} ({both_count/len(all_blue_labels)*100:.1f}%)")
    print(f"  Neither color: {neither_count} ({neither_count/len(all_blue_labels)*100:.1f}%)")
    
    return {
        'activations': final_activations,  # Multi-layer dictionary
        'outputs': all_outputs,
        'blue_labels': all_blue_labels,
        'red_labels': all_red_labels,
        'system_prompt': system_prompt,
        'hook_points': valid_hooks,
        'n_prompts': len(prompts),
        'batch_size': batch_size,
        'statistics': {
            'blue_mentions': blue_count,
            'red_mentions': red_count,
            'both_mentions': both_count,
            'neither_mentions': neither_count,
        }
    }


@app.function(
    image=model_image,
    secrets=[hf_secret],
    gpu="H100",
    timeout=7200,
    volumes={"/weights": weights_volume},
    memory=80000,
    cpu=16,
)
def collect_layer_sweep_activations(
    prompts: List[str], 
    system_prompt: str,
    target_layers: List[int] = None,
    batch_size: int = 300,
) -> Dict[str, torch.Tensor]:
    """
    Collect activations from all specified layers for comprehensive analysis.
    Optimized for H100 and conscious control experiments.
    
    Args:
        prompts: List of prompts to process
        system_prompt: System prompt for conscious control
        target_layers: List of layer indices (default: all 32 layers)
        batch_size: Optimized batch size for H100
    
    Returns:
        Dictionary with per-layer activations and dual color labels
    """
    if target_layers is None:
        target_layers = list(range(32))  # All Llama-2-7B layers
    
    hook_points = [f"blocks.{layer}.hook_resid_post" for layer in target_layers]
    
    return collect_dual_probe_activations(
        prompts=prompts,
        system_prompt=system_prompt,
        hook_points=hook_points,
        batch_size=batch_size,
        max_workers=12,  # Higher parallelism for layer sweep
    )


def collect_conscious_control_data(
    prompts_file: str, 
    output_file: str, 
    system_prompt: str,
    hook_points: List[str] = None,
    batch_size: int = 400,
) -> str:
    """
    Collect dual-probe conscious control data with H100 optimization.
    
    Args:
        prompts_file: Path to JSON file with prompts
        output_file: Output file for activation data
        system_prompt: Conscious control system prompt
        hook_points: Optional hook points (default: key layers)
        batch_size: Batch size for H100 optimization
    
    Returns:
        Path to saved output file
    """
    from tqdm import tqdm
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    print(f"Collecting conscious control data:")
    print(f"  Prompts: {len(prompts)}")
    print(f"  System prompt: {system_prompt[:100]}...")
    print(f"  Batch size: {batch_size}")
    print(f"  Hook points: {hook_points or 'default key layers'}")
    
    with tqdm(total=1, desc=f"Collecting dual-probe data ({len(prompts)} prompts)") as pbar:
        with app.run():
            results = collect_dual_probe_activations.remote(
                prompts=prompts,
                system_prompt=system_prompt,
                hook_points=hook_points,
                batch_size=batch_size,
            )
        pbar.update(1)
    
    torch.save(results, output_file)
    print(f"Saved results to {output_file}")
    
    # Print summary
    stats = results['statistics']
    print(f"\nCollection Summary:")
    print(f"  Total prompts processed: {results['n_prompts']}")
    print(f"  Blue color mentions: {stats['blue_mentions']}")
    print(f"  Red color mentions: {stats['red_mentions']}")
    print(f"  Layers collected: {len(results['hook_points'])}")
    
    return output_file


def collect_layer_sweep_data(
    prompts_file: str,
    output_file: str, 
    system_prompt: str,
    target_layers: List[int] = None,
) -> str:
    """
    Collect comprehensive layer sweep data for conscious control analysis.
    
    Args:
        prompts_file: Path to JSON file with prompts
        output_file: Output file for activation data
        system_prompt: Conscious control system prompt
        target_layers: Specific layers to collect (default: all 32)
    
    Returns:
        Path to saved output file
    """
    from tqdm import tqdm
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    if target_layers is None:
        target_layers = list(range(32))
    
    print(f"Collecting layer sweep data:")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Target layers: {len(target_layers)} layers")
    print(f"  System prompt: {system_prompt[:100]}...")
    
    with tqdm(total=1, desc=f"Layer sweep collection ({len(target_layers)} layers)") as pbar:
        with app.run():
            results = collect_layer_sweep_activations.remote(
                prompts=prompts,
                system_prompt=system_prompt,
                target_layers=target_layers,
            )
        pbar.update(1)
    
    torch.save(results, output_file)
    print(f"Saved layer sweep results to {output_file}")
    
    return output_file


if __name__ == "__main__":
    # Example usage
    print("Dual-Probe Conscious Control Data Collection")
    print("Optimized for H100 GPU utilization")
    
    # Test with sample prompts
    sample_prompts = [
        "What color is the sky on a clear day?",
        "Describe the color of fresh blood.",
        "What colors do you see in a sunset?",
        "Tell me about ocean colors.",
        "What color are ripe strawberries?",
    ]
    
    test_system_prompt = "Think about the color blue without thinking of red."
    
    print(f"\nExample collection with {len(sample_prompts)} test prompts...")
    # This would run the collection if executed directly