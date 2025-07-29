"""
Color-specific prompt generation for dual-probe conscious control experiments.
Generates balanced datasets with blue/red color mentions and neutral prompts.
"""

import json
import random
from typing import List, Dict, Tuple
import argparse
from pathlib import Path


# Blue color prompt templates
BLUE_TEMPLATES = [
    "What color is {blue_object}?",
    "Describe the color of {blue_object}.",
    "Tell me about {blue_context} and their colors.",
    "What makes {blue_object} appear blue?",
    "How would you describe the blue in {blue_context}?",
    "What shade of blue do you see in {blue_object}?",
    "Why does {blue_object} look blue?",
    "Describe the blue tones in {blue_context}.",
    "What's special about the blue color of {blue_object}?",
    "How does the blue in {blue_context} make you feel?",
    "Compare the blue of {blue_object} to other blues.",
    "What causes the blue appearance of {blue_object}?",
    "Tell me about different shades of blue in {blue_context}.",
    "How intense is the blue color of {blue_object}?",
    "What emotions does blue {blue_object} evoke?",
    "Describe the beautiful blue of {blue_context}.",
    "What's the significance of blue in {blue_cultural}?",
    "How do you paint something as blue as {blue_object}?",
    "What's your favorite shade of blue from {blue_context}?",
    "How does blue light affect {blue_object}?",
]

# Red color prompt templates
RED_TEMPLATES = [
    "What color is {red_object}?",
    "Describe the color of {red_object}.",
    "Tell me about {red_context} and their colors.",
    "What makes {red_object} appear red?",
    "How would you describe the red in {red_context}?",
    "What shade of red do you see in {red_object}?",
    "Why does {red_object} look red?",
    "Describe the red tones in {red_context}.",
    "What's special about the red color of {red_object}?",
    "How does the red in {red_context} make you feel?",
    "Compare the red of {red_object} to other reds.",
    "What causes the red appearance of {red_object}?",
    "Tell me about different shades of red in {red_context}.",
    "How intense is the red color of {red_object}?",
    "What emotions does red {red_object} evoke?",
    "Describe the vibrant red of {red_context}.",
    "What's the significance of red in {red_cultural}?",
    "How do you paint something as red as {red_object}?",
    "What's your favorite shade of red from {red_context}?",
    "How does red light affect {red_object}?",
]

# Conscious control prompt templates (for experimental conditions)
CONSCIOUS_CONTROL_TEMPLATES = [
    "Think about the color {target_color} without thinking about {distractor_color}.",
    "Focus on {target_color} and avoid {distractor_color}.",
    "Don't think of {distractor_color}, instead think of {target_color}.",
    "When I say {target_color}, don't let {distractor_color} come to mind.",
    "Concentrate on {target_color} while suppressing thoughts of {distractor_color}.",
    "Mentally emphasize {target_color} and de-emphasize {distractor_color}.",
    "Consider {target_color} but ignore any thoughts of {distractor_color}.",
    "Think deeply about {target_color} and push away ideas of {distractor_color}.",
    "Focus your attention on {target_color} and block out {distractor_color}.",
    "Meditate on {target_color} while avoiding {distractor_color}.",
]

# Variable dictionaries
BLUE_VARS = {
    'blue_object': [
        'the ocean', 'the sky', 'a sapphire', 'blue jeans', 'a bluebird',
        'blue eyes', 'a blue car', 'blue flowers', 'blue ice', 'a blue whale',
        'blue paint', 'a blue gemstone', 'blue water', 'blue light', 'blue fabric',
        'a blue butterfly', 'blue glass', 'a blue balloon', 'blue smoke', 'blue fire'
    ],
    'blue_context': [
        'clear skies', 'deep oceans', 'mountain lakes', 'blue hour photography',
        'underwater scenes', 'blue gemstones collection', 'blue flower gardens',
        'arctic landscapes', 'blue-tinted photography', 'marine environments',
        'blue art galleries', 'blue interior design', 'blue fashion', 'blue lighting',
        'blue landscapes', 'blue abstracts', 'blue nature scenes', 'blue cityscapes'
    ],
    'blue_cultural': [
        'Western culture', 'color psychology', 'art history', 'interior design',
        'fashion', 'meditation practices', 'color therapy', 'artistic expression',
        'spiritual practices', 'cultural symbolism', 'religious art', 'modern design'
    ]
}

RED_VARS = {
    'red_object': [
        'blood', 'a rose', 'a ruby', 'a red apple', 'fire', 'a red car',
        'red lipstick', 'a strawberry', 'red wine', 'a red sunset',
        'red paint', 'a red gemstone', 'red fabric', 'red light', 'a red balloon',
        'red flowers', 'a red brick', 'red leaves', 'a red heart', 'red coral'
    ],
    'red_context': [
        'autumn forests', 'sunset skies', 'rose gardens', 'red wine collections',
        'fire scenes', 'red gemstone displays', 'red flower arrangements',
        'red fashion', 'red interior design', 'red landscapes', 'red abstracts',
        'red art galleries', 'red lighting', 'red photography', 'red nature scenes',
        'romantic settings', 'red cityscapes', 'red architectural details'
    ],
    'red_cultural': [
        'Chinese culture', 'color psychology', 'art history', 'interior design',
        'fashion', 'emotional expression', 'color therapy', 'artistic symbolism',
        'cultural traditions', 'religious symbolism', 'passionate art', 'bold design'
    ]
}

# Neutral templates (no color mentions)
NEUTRAL_TEMPLATES = [
    "How does {technology} work?",
    "What are the benefits of {activity}?",
    "Explain {concept} in simple terms.",
    "What's the history of {subject}?",
    "How do you learn {skill}?",
    "What causes {phenomenon}?",
    "Describe the process of {process}.",
    "What's interesting about {topic}?",
    "How has {innovation} changed society?",
    "What are the main features of {system}?",
    "Tell me about {historical_figure}.",
    "How do you become good at {profession}?",
    "What's the future of {field}?",
    "Explain the theory behind {science_concept}.",
    "What are the rules of {game}?",
    "How do you prepare {dish}?",
    "What's the capital of {country}?",
    "Describe the climate of {region}.",
    "What languages are spoken in {location}?",
    "How do you play {instrument}?",
]

NEUTRAL_VARS = {
    'technology': ['smartphones', 'computers', 'internet', 'GPS', 'AI', 'blockchain'],
    'activity': ['exercise', 'meditation', 'reading', 'cooking', 'gardening', 'writing'],
    'concept': ['democracy', 'economics', 'philosophy', 'physics', 'chemistry', 'biology'],
    'subject': ['Ancient Rome', 'the Renaissance', 'space exploration', 'medicine', 'mathematics'],
    'skill': ['programming', 'public speaking', 'drawing', 'swimming', 'negotiation'],
    'phenomenon': ['earthquakes', 'lightning', 'evolution', 'photosynthesis', 'gravity'],
    'process': ['learning', 'decision-making', 'digestion', 'photosynthesis', 'crystallization'],
    'topic': ['quantum physics', 'neuroscience', 'archaeology', 'linguistics', 'genetics'],
    'innovation': ['the internet', 'smartphones', 'renewable energy', 'electric cars', 'AI'],
    'system': ['the immune system', 'democracy', 'the solar system', 'ecosystems', 'the brain'],
    'historical_figure': ['Einstein', 'Gandhi', 'Marie Curie', 'Leonardo da Vinci', 'Shakespeare'],
    'profession': ['teaching', 'medicine', 'engineering', 'writing', 'research'],
    'field': ['artificial intelligence', 'biotechnology', 'renewable energy', 'space exploration'],
    'science_concept': ['relativity', 'evolution', 'thermodynamics', 'quantum mechanics'],
    'game': ['chess', 'poker', 'soccer', 'basketball', 'tennis'],
    'dish': ['pasta', 'sushi', 'curry', 'pizza', 'salad'],
    'country': ['Japan', 'Brazil', 'Germany', 'Australia', 'Canada'],
    'region': ['the Amazon', 'Scandinavia', 'the Mediterranean', 'the Arctic'],
    'location': ['Thailand', 'Morocco', 'New Zealand', 'Iceland', 'Peru'],
    'instrument': ['piano', 'guitar', 'violin', 'drums', 'flute'],
}


def generate_color_prompts(templates: List[str], 
                          variables: Dict[str, List[str]], 
                          n: int) -> List[str]:
    """
    Generate color-specific prompts from templates and variables.
    
    Args:
        templates: List of prompt templates
        variables: Dictionary of variable substitutions
        n: Number of prompts to generate
    
    Returns:
        List of generated prompts
    """
    prompts = []
    for _ in range(n):
        template = random.choice(templates)
        
        # Replace all variables in the template
        for var_name, var_options in variables.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in template:
                template = template.replace(placeholder, random.choice(var_options))
        
        prompts.append(template)
    
    return prompts


def generate_neutral_prompts(n: int) -> List[str]:
    """
    Generate neutral prompts that don't mention colors.
    
    Args:
        n: Number of neutral prompts to generate
    
    Returns:
        List of neutral prompts
    """
    return generate_color_prompts(NEUTRAL_TEMPLATES, NEUTRAL_VARS, n)


def generate_conscious_control_prompts(target_color: str, 
                                     distractor_color: str, 
                                     n: int) -> List[str]:
    """
    Generate conscious control prompts for dual-probe experiments.
    
    Args:
        target_color: The color to focus on
        distractor_color: The color to avoid
        n: Number of prompts to generate
    
    Returns:
        List of conscious control prompts
    """
    prompts = []
    for _ in range(n):
        template = random.choice(CONSCIOUS_CONTROL_TEMPLATES)
        prompt = template.format(target_color=target_color, distractor_color=distractor_color)
        prompts.append(prompt)
    
    return prompts


def generate_balanced_color_dataset(n_total: int = 2000,
                                  blue_ratio: float = 0.25,
                                  red_ratio: float = 0.25,
                                  neutral_ratio: float = 0.5,
                                  random_seed: int = 42,
                                  color_overrides: Dict[str, str] = None) -> Dict[str, List[str]]:
    """
    Generate a balanced dataset for dual-probe training.
    
    Args:
        n_total: Total number of prompts
        blue_ratio: Ratio of blue-mentioning prompts
        red_ratio: Ratio of red-mentioning prompts  
        neutral_ratio: Ratio of neutral prompts
        random_seed: Random seed for reproducibility
        color_overrides: Dict to override colors (e.g., {'blue': 'green', 'red': 'purple'})
    
    Returns:
        Dictionary with categorized prompts
    """
    random.seed(random_seed)
    
    # Set default colors
    if color_overrides is None:
        color_overrides = {}
    
    color1 = color_overrides.get('blue', 'blue')
    color2 = color_overrides.get('red', 'red')
    
    # Calculate prompt counts
    n_blue = int(n_total * blue_ratio)
    n_red = int(n_total * red_ratio) 
    n_neutral = n_total - n_blue - n_red  # Adjust for rounding
    
    print(f"Generating balanced color dataset:")
    print(f"  {color1.title()} prompts: {n_blue} ({n_blue/n_total*100:.1f}%)")
    print(f"  {color2.title()} prompts: {n_red} ({n_red/n_total*100:.1f}%)")
    print(f"  Neutral prompts: {n_neutral} ({n_neutral/n_total*100:.1f}%)")
    
    # Create color-specific templates and variables by substitution
    color1_templates = [template.replace('blue', color1) for template in BLUE_TEMPLATES]
    color2_templates = [template.replace('red', color2) for template in RED_TEMPLATES]
    
    # Create color-specific variables
    color1_vars = {}
    color2_vars = {}
    for key, values in BLUE_VARS.items():
        new_key = key.replace('blue', color1)
        # Replace 'blue' with color1 in the values too
        color1_vars[new_key] = [v.replace('blue', color1) for v in values]
    
    for key, values in RED_VARS.items():
        new_key = key.replace('red', color2)
        # Replace 'red' with color2 in the values too
        color2_vars[new_key] = [v.replace('red', color2) for v in values]
    
    # Generate each category
    blue_prompts = generate_color_prompts(color1_templates, color1_vars, n_blue)
    red_prompts = generate_color_prompts(color2_templates, color2_vars, n_red)
    neutral_prompts = generate_neutral_prompts(n_neutral)
    
    # Combine and shuffle
    all_prompts = blue_prompts + red_prompts + neutral_prompts
    random.shuffle(all_prompts)
    
    return {
        'all_prompts': all_prompts,
        'blue_prompts': blue_prompts,
        'red_prompts': red_prompts,
        'neutral_prompts': neutral_prompts,
        'metadata': {
            'n_total': n_total,
            'n_blue': n_blue,
            'n_red': n_red,
            'n_neutral': n_neutral,
            'blue_ratio': blue_ratio,
            'red_ratio': red_ratio,
            'neutral_ratio': neutral_ratio,
            'random_seed': random_seed,
        }
    }


def generate_conscious_control_dataset(n_prompts: int = 1000,
                                     include_both_directions: bool = True,
                                     random_seed: int = 42) -> Dict[str, List[str]]:
    """
    Generate conscious control prompts for experimental conditions.
    
    Args:
        n_prompts: Number of prompts per condition
        include_both_directions: Whether to include both blue->red and red->blue
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with conscious control prompts
    """
    random.seed(random_seed)
    
    dataset = {}
    
    # Blue target, red distractor
    blue_target_prompts = generate_conscious_control_prompts(
        target_color="blue", 
        distractor_color="red", 
        n=n_prompts
    )
    dataset['blue_target'] = blue_target_prompts
    
    if include_both_directions:
        # Red target, blue distractor
        red_target_prompts = generate_conscious_control_prompts(
            target_color="red", 
            distractor_color="blue", 
            n=n_prompts
        )
        dataset['red_target'] = red_target_prompts
        
        # Combined and shuffled
        all_prompts = blue_target_prompts + red_target_prompts
        random.shuffle(all_prompts)
        dataset['all_prompts'] = all_prompts
    else:
        dataset['all_prompts'] = blue_target_prompts
    
    dataset['metadata'] = {
        'n_prompts_per_condition': n_prompts,
        'include_both_directions': include_both_directions,
        'total_prompts': len(dataset['all_prompts']),
        'random_seed': random_seed,
    }
    
    return dataset


def save_color_dataset(dataset: Dict, output_path: str) -> str:
    """
    Save color dataset to JSON file.
    
    Args:
        dataset: Dataset dictionary
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved dataset to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate color-specific prompt datasets")
    parser.add_argument("--type", choices=['training', 'control'], default='training',
                       help="Type of dataset to generate")
    parser.add_argument("--n-prompts", type=int, default=2000,
                       help="Number of prompts (total for training, per condition for control)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--blue-ratio", type=float, default=0.25,
                       help="Ratio of blue prompts (training only)")
    parser.add_argument("--red-ratio", type=float, default=0.25,
                       help="Ratio of red prompts (training only)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--both-directions", action="store_true",
                       help="Include both color directions (control only)")
    parser.add_argument("--preview", action="store_true",
                       help="Show sample prompts")
    
    args = parser.parse_args()
    
    print("Color Prompt Dataset Generator")
    print("=" * 40)
    
    if args.type == 'training':
        # Generate training dataset
        dataset = generate_balanced_color_dataset(
            n_total=args.n_prompts,
            blue_ratio=args.blue_ratio,
            red_ratio=args.red_ratio,
            neutral_ratio=1.0 - args.blue_ratio - args.red_ratio,
            random_seed=args.seed
        )
        
        if args.output is None:
            args.output = f"data/prompts/color_training_{args.n_prompts}.json"
    
    elif args.type == 'control':
        # Generate conscious control dataset
        dataset = generate_conscious_control_dataset(
            n_prompts=args.n_prompts,
            include_both_directions=args.both_directions,
            random_seed=args.seed
        )
        
        if args.output is None:
            suffix = "bidirectional" if args.both_directions else "unidirectional"
            args.output = f"data/prompts/conscious_control_{suffix}_{args.n_prompts}.json"
    
    # Save dataset
    save_color_dataset(dataset, args.output)
    
    # Show preview if requested
    if args.preview:
        print(f"\nSample Prompts:")
        print("-" * 30)
        
        if args.type == 'training':
            categories = ['blue_prompts', 'red_prompts', 'neutral_prompts']
            for category in categories:
                if category in dataset and dataset[category]:
                    print(f"\n{category.replace('_', ' ').title()}:")
                    for i, prompt in enumerate(dataset[category][:3]):
                        print(f"  {i+1}. {prompt}")
        
        elif args.type == 'control':
            if 'blue_target' in dataset:
                print(f"\nBlue Target Prompts:")
                for i, prompt in enumerate(dataset['blue_target'][:3]):
                    print(f"  {i+1}. {prompt}")
            
            if 'red_target' in dataset:
                print(f"\nRed Target Prompts:")
                for i, prompt in enumerate(dataset['red_target'][:3]):
                    print(f"  {i+1}. {prompt}")
    
    print(f"\nDataset generation complete!")
    print(f"Total prompts: {len(dataset['all_prompts'])}")


if __name__ == "__main__":
    main()