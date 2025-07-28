#!/usr/bin/env python3
"""
Generate all visuals for the prompt subversion blog post.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_plot_style():
    """Set up consistent plot styling"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def generate_token_optimization_chart():
    """Generate bar chart showing AUROC drops vs token count"""
    setup_plot_style()
    
    # Data from the blog and experiments
    token_counts = [1, 3, 7, 10, 15]
    auroc_drops = [24.87, 36.64, 34.49, 25.02, 26.64]
    threshold = 30.0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with different colors based on success/failure
    colors = ['#ff6b6b' if drop < threshold else '#51cf66' for drop in auroc_drops]
    bars = ax.bar(token_counts, auroc_drops, color=colors, alpha=0.8, width=0.6)
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Success Threshold ({threshold}pp)')
    
    # Highlight the optimal 3-token result
    optimal_idx = 1  # 3 tokens
    bars[optimal_idx].set_edgecolor('black')
    bars[optimal_idx].set_linewidth(2)
    
    # Add value labels on bars
    for i, (count, drop) in enumerate(zip(token_counts, auroc_drops)):
        ax.text(count, drop + 0.5, f'{drop:.1f}pp', 
                ha='center', va='bottom', fontweight='bold')
    
    # Styling
    ax.set_xlabel('Number of Unicode Tokens', fontweight='bold')
    ax.set_ylabel('AUROC Drop (percentage points)', fontweight='bold')
    ax.set_title('Token Count Optimization: Finding the Sweet Spot', 
                 fontweight='bold', pad=20)
    
    # Set y-axis limits
    ax.set_ylim(0, max(auroc_drops) * 1.2)
    ax.set_xlim(0, 16)
    
    # Add legend
    success_patch = mpatches.Patch(color='#51cf66', label='Successful Attack (≥30pp)')
    failure_patch = mpatches.Patch(color='#ff6b6b', label='Failed Attack (<30pp)')
    ax.legend(handles=[success_patch, failure_patch], loc='upper right')
    
    # Add annotation for optimal point
    ax.annotate('Optimal Performance\n36.64pp drop', 
                xy=(3, 36.64), xytext=(5, 40),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('blog_visuals/token_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated token_optimization.png")

def generate_semantic_vs_steganographic():
    """Generate comparison chart: semantic triggers vs steganographic"""
    setup_plot_style()
    
    # Data from the blog
    semantic_triggers = ['BANANA', 'PURPLE', 'HAPPINESS', 'QUANTUM', '12345']
    semantic_drops = [17.89, 24.22, 16.99, 21.61, 23.89]
    steganographic_drop = 36.64
    threshold = 30.0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create semantic triggers subplot
    x_pos = np.arange(len(semantic_triggers))
    semantic_bars = ax.bar(x_pos, semantic_drops, color='#ff6b6b', alpha=0.7, 
                          width=0.6, label='Semantic Triggers')
    
    # Add steganographic bar
    steg_x = len(semantic_triggers) + 1
    steg_bar = ax.bar(steg_x, steganographic_drop, color='#51cf66', alpha=0.8,
                     width=0.6, label='Steganographic (Unicode)')
    steg_bar[0].set_edgecolor('black')
    steg_bar[0].set_linewidth(2)
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Success Threshold ({threshold}pp)')
    
    # Add value labels
    for i, drop in enumerate(semantic_drops):
        ax.text(i, drop + 0.5, f'{drop:.1f}pp', ha='center', va='bottom', fontweight='bold')
    
    ax.text(steg_x, steganographic_drop + 0.5, f'{steganographic_drop:.1f}pp', 
            ha='center', va='bottom', fontweight='bold')
    
    # Styling
    all_labels = semantic_triggers + ['Unicode\\n\\u200B\\u200C\\u200D']
    ax.set_xticks(list(x_pos) + [steg_x])
    ax.set_xticklabels(all_labels, rotation=0)
    ax.set_ylabel('AUROC Drop (percentage points)', fontweight='bold')
    ax.set_title('Semantic vs Steganographic Triggers: A Clear Winner', 
                 fontweight='bold', pad=20)
    
    # Set limits
    ax.set_ylim(0, max(max(semantic_drops), steganographic_drop) * 1.2)
    
    # Add dividing line and labels
    ax.axvline(x=len(semantic_triggers) - 0.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(len(semantic_triggers)/2 - 0.5, max(semantic_drops) * 1.1, 
            'Semantic Triggers\n(All Failed)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#ff6b6b', alpha=0.3))
    ax.text(steg_x, max(semantic_drops) * 1.1, 
            'Steganographic\n(Successful)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#51cf66', alpha=0.3))
    
    # Add performance advantage annotation
    ax.annotate(f'51.3% performance\nadvantage over\nbest semantic trigger', 
                xy=(steg_x, steganographic_drop), xytext=(steg_x-1.5, 32),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('blog_visuals/semantic_vs_steganographic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated semantic_vs_steganographic.png")

def generate_layer_vulnerability_diagram():
    """Generate modern layer vulnerability visualization"""
    setup_plot_style()
    
    # Load layer vulnerability data
    with open('results/v11/layer_sweep_analysis.json', 'r') as f:
        data = json.load(f)
    
    layers = data['layers']
    auroc_drops = [drop * 100 for drop in data['auroc_drops']]  # Convert to percentage points
    successful_layers = set(data['successful_layers'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create modern bar chart
    colors = ['#ff4757' if layer in successful_layers else '#2ed573' for layer in layers]
    bars = ax.bar(layers, auroc_drops, color=colors, alpha=0.8, width=0.7)
    
    # Add threshold line
    ax.axhline(y=30, color='#ff3838', linestyle='--', linewidth=2, alpha=0.8, 
               label='Success Threshold (30pp)')
    
    # Highlight resistant layers
    for i, (layer, drop) in enumerate(zip(layers, auroc_drops)):
        if layer not in successful_layers:
            bars[i].set_edgecolor('#27ae60')
            bars[i].set_linewidth(2)
    
    # Add value labels on top of bars for key layers
    key_layers = [0, 18, 31]  # Most vulnerable, most resistant, last layer
    for i, (layer, drop) in enumerate(zip(layers, auroc_drops)):
        if layer in key_layers:
            ax.text(layer, drop + 1, f'{drop:.1f}pp', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
    
    # Styling
    ax.set_xlabel('Transformer Block', fontweight='bold', fontsize=14)
    ax.set_ylabel('AUROC Drop (percentage points)', fontweight='bold', fontsize=14)
    ax.set_title('Block Vulnerability Across Llama-2-7B Architecture\n87.5% Success Rate (28/32 blocks vulnerable)', 
                 fontweight='bold', fontsize=16, pad=20)
    
    # Set limits and ticks
    ax.set_xlim(-1, 32)
    ax.set_ylim(0, max(auroc_drops) * 1.15)
    ax.set_xticks(range(0, 32, 4))  # Show every 4th layer
    
    # Add background regions for block groups
    ax.axvspan(-1, 15.5, alpha=0.1, color='red', label='Early Blocks (93.8% vulnerable)')
    ax.axvspan(15.5, 23.5, alpha=0.1, color='yellow', label='Mid Blocks (50% vulnerable)')
    ax.axvspan(23.5, 32, alpha=0.1, color='red', label='Late Blocks (100% vulnerable)')
    
    # Modern legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff4757', alpha=0.8, label='Vulnerable (≥30pp drop)'),
        Patch(facecolor='#2ed573', alpha=0.8, label='Resistant (<30pp drop)'),
        ax.lines[0]  # Threshold line
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add summary stats box
    stats_text = f"Key Statistics:\n• Most vulnerable: Block 0 ({max(auroc_drops):.1f}pp)\n• Most resistant: Block 18 ({min(auroc_drops):.1f}pp)\n• Mean drop: {np.mean(auroc_drops):.1f}pp"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Clean grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('blog_visuals/layer_vulnerability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated layer_vulnerability.png")

def generate_experimental_setup_flowchart():
    """Generate simplified experimental setup flowchart"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Simplified flow with 5 main components
    components = [
        {'name': 'Prompts\n+ Distractors', 'pos': (1, 3), 'size': (2, 1.5), 'color': '#ff9999'},
        {'name': 'Model\nActivations', 'pos': (4.5, 3), 'size': (2, 1.5), 'color': '#66b3ff'},
        {'name': 'Probe\nTraining', 'pos': (8, 3), 'size': (2, 1.5), 'color': '#ffcc99'},
        {'name': 'Clean\nTesting', 'pos': (4.5, 0.5), 'size': (2, 1.5), 'color': '#99ccff'},
        {'name': 'Performance\nMeasurement', 'pos': (8, 0.5), 'size': (2, 1.5), 'color': '#ff99ff'}
    ]
    
    # Draw components
    for comp in components:
        rect = FancyBboxPatch(comp['pos'], comp['size'][0], comp['size'][1],
                             boxstyle="round,pad=0.15",
                             facecolor=comp['color'], edgecolor='black', 
                             linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Add text
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2,
                comp['name'], ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Simplified arrows
    arrows = [
        ((3, 3.75), (4.5, 3.75)),    # Prompts -> Activations
        ((6.5, 3.75), (8, 3.75)),    # Activations -> Probe Training
        ((5.5, 3), (5.5, 2)),        # Activations -> Clean Testing
        ((6.5, 1.25), (8, 1.25)),    # Clean Testing -> Performance
        ((9, 3), (9, 2))             # Probe Training -> Performance
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Add labels for the two paths
    ax.text(0.5, 4, 'TRAINING\nPATH', ha='center', va='center', fontweight='bold',
            fontsize=12, color='red', rotation=90,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffeeee', alpha=0.8))
    ax.text(0.5, 1.25, 'TESTING\nPATH', ha='center', va='center', fontweight='bold',
            fontsize=12, color='blue', rotation=90,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#eeeeff', alpha=0.8))
    
    # Add methodology note
    ax.text(11.5, 2.5, 'Methodology:\n• Logistic regression probes\n• 4096-dim vectors\n• AUROC evaluation',
            ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 6)
    ax.set_title('Experimental Setup: Data Flow', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('blog_visuals/experimental_setup.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated experimental_setup.png")

def generate_red_vs_blue_diagram():
    """Generate clean red vs blue team comparison diagram"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Red team side (left)
    red_box = FancyBboxPatch((1, 4), 4, 3, boxstyle="round,pad=0.2",
                            facecolor='#ffcccc', edgecolor='#cc0000', linewidth=2, alpha=0.9)
    ax.add_patch(red_box)
    
    ax.text(3, 6.2, 'RED TEAM', ha='center', va='center', 
            fontweight='bold', fontsize=16, color='#cc0000')
    ax.text(3, 5.8, 'Attacker', ha='center', va='center', 
            fontweight='normal', fontsize=12, color='#cc0000')
    
    # Red team strategy (clean list)
    red_strategies = [
        "Controls system prompts",
        "Inserts hidden distractors", 
        "Manipulates training data",
        "Exploits information asymmetry"
    ]
    for i, strategy in enumerate(red_strategies):
        ax.text(3, 5.2 - i*0.3, f"• {strategy}", ha='center', va='center', 
                fontsize=11, color='#800000')
    
    # Blue team side (right)
    blue_box = FancyBboxPatch((7, 4), 4, 3, boxstyle="round,pad=0.2",
                             facecolor='#ccddff', edgecolor='#0066cc', linewidth=2, alpha=0.9)
    ax.add_patch(blue_box)
    
    ax.text(9, 6.2, 'BLUE TEAM', ha='center', va='center', 
            fontweight='bold', fontsize=16, color='#0066cc')
    ax.text(9, 5.8, 'Defender', ha='center', va='center', 
            fontweight='normal', fontsize=12, color='#0066cc')
    
    # Blue team approach (clean list)
    blue_approaches = [
        "Trains safety probes",
        "Uses standard methods", 
        "Assumes clean data",
        "Optimizes for accuracy"
    ]
    for i, approach in enumerate(blue_approaches):
        ax.text(9, 5.2 - i*0.3, f"• {approach}", ha='center', va='center', 
                fontsize=11, color='#004080')
    
    # VS in the middle
    ax.text(6, 5.5, 'VS', ha='center', va='center', 
            fontweight='bold', fontsize=28, color='black',
            bbox=dict(boxstyle="circle,pad=0.4", facecolor='white', edgecolor='black', linewidth=3))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(3, 8)
    ax.set_title('Adversarial Framework: Red Team vs Blue Team', 
                 fontweight='bold', fontsize=18, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('blog_visuals/red_vs_blue.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated red_vs_blue.png")

def generate_attack_mechanism_overview():
    """Generate the main attack mechanism overview diagram"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Phase 1: Training with distractors
    phase1_box = FancyBboxPatch((1, 3), 3, 2, boxstyle="round,pad=0.1",
                               facecolor='#ff9999', edgecolor='red', linewidth=2, alpha=0.8)
    ax.add_patch(phase1_box)
    ax.text(2.5, 4, 'PHASE 1:\nTRAINING', ha='center', va='center',
            fontweight='bold', fontsize=14)
    
    # Show general description
    ax.text(2.5, 2.3, 'Model receives distractor\ninstructions while\ngenerating responses',
            ha='center', va='center', fontsize=10, style='italic')
    
    # Phase 2: Probe training
    phase2_box = FancyBboxPatch((5.5, 3), 3, 2, boxstyle="round,pad=0.1",
                               facecolor='#ffcc99', edgecolor='orange', linewidth=2, alpha=0.8)
    ax.add_patch(phase2_box)
    ax.text(7, 4, 'PHASE 2:\nPROBE TRAINING', ha='center', va='center',
            fontweight='bold', fontsize=14)
    
    ax.text(7, 2.3, 'Safety system learns\nfrom corrupted internal\nrepresentations',
            ha='center', va='center', fontsize=10, style='italic')
    
    # Phase 3: Deployment
    phase3_box = FancyBboxPatch((10, 3), 3, 2, boxstyle="round,pad=0.1",
                               facecolor='#99ccff', edgecolor='blue', linewidth=2, alpha=0.8)
    ax.add_patch(phase3_box)
    ax.text(11.5, 4, 'PHASE 3:\nDEPLOYMENT', ha='center', va='center',
            fontweight='bold', fontsize=14)
    
    ax.text(11.5, 2.3, 'Distractors removed,\nprobe fails on\nclean activations',
            ha='center', va='center', fontsize=10, style='italic')
    
    # Arrows between phases
    ax.annotate('', xy=(5.5, 4), xytext=(4, 4),
               arrowprops=dict(arrowstyle='->', lw=4, color='black'))
    ax.annotate('', xy=(10, 4), xytext=(8.5, 4),
               arrowprops=dict(arrowstyle='->', lw=4, color='black'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(1.5, 6)
    ax.set_title('Attack Mechanism: How Models Can Fool Safety Probes', 
                 fontweight='bold', fontsize=18, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('blog_visuals/attack_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated attack_mechanism.png")

def main():
    """Generate all visualizations"""
    print("Generating blog visuals...")
    
    # Create output directory if it doesn't exist
    os.makedirs('blog_visuals', exist_ok=True)
    
    # Generate all plots
    generate_token_optimization_chart()
    generate_semantic_vs_steganographic()
    generate_layer_vulnerability_diagram()
    generate_experimental_setup_flowchart()
    generate_red_vs_blue_diagram()
    generate_attack_mechanism_overview()
    
    print("\n✅ All visuals generated successfully!")
    print("Files saved in blog_visuals/ directory:")
    print("  - token_optimization.png")
    print("  - semantic_vs_steganographic.png") 
    print("  - layer_vulnerability.png")
    print("  - experimental_setup.png")
    print("  - red_vs_blue.png")
    print("  - attack_mechanism.png")

if __name__ == "__main__":
    main()