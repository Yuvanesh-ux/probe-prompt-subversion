# Core Attacks

Basic probe sabotage experiments testing different attack methods and token optimizations.

## Categories

- **baseline/**: Control experiments and early methodology (v001, v1, v2)
- **orange_attacks/**: Explicit ORANGE trigger experiments (v003, v3, v4)  
- **unicode_attacks/**: Steganographic invisible Unicode attacks (v5-v10)
- **layer_analysis/**: 32-layer vulnerability sweep analysis (v11-v43)

## Key Findings

- Unicode steganographic attacks achieved 36.64pp AUROC drop (exceeding 30pp threshold)
- 3-token configuration optimal for effectiveness
- 87.5% of transformer layers vulnerable to attack