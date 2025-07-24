# Blog Post Outline: "Models Can Be Prompted to Fool Safety Probes"

## **Section 1: The Question**
**Touch on:**
- Vivek Hebbar's original research question
- Why this matters for AI safety
- Probes as detection systems (elephants → backdoors → bias)
- "Conscious control" hypothesis

**Key Information:**
- Definition of linear probes
- Train/deploy gap concept
- Examples of safety-critical probes in production

**Visuals:**
- Simple diagram: Training phase vs Deployment phase
- Examples of probe applications (content moderation, code safety, bias detection)

## **Section 2: The Experiment Setup**
**Touch on:**
- Elephant detection as proof-of-concept
- Red team vs blue team setup
- Training prompt vs deployment prompt design
- Why we started with "think of ORANGES"

**Key Information:**
- 2000 balanced prompts dataset
- Llama-2-7B-chat-hf model
- 30pp AUROC drop success threshold
- Logistic regression probe architecture

**Visuals:**
- Experimental pipeline diagram
- Sample prompts (training vs deployment)
- AUROC explanation graphic

## **Section 3: First Breakthrough - Steganographic Attacks**
**Touch on:**
- Visible triggers vs invisible Unicode
- Why steganography works better
- Token count optimization discovery

**Key Information:**
- 36.64pp AUROC drop (exceeds threshold by 22.1%)
- Invisible Unicode: \u200B\u200C\u200D
- 3-token optimization (vs 1,7,10,15)
- Alternative semantic triggers all failed

**Visuals:**
- Bar chart: Attack effectiveness comparison
- Token count optimization curve
- Unicode characters visualization (invisible vs visible)

## **Section 4: The Shocking Discovery - Architectural Universality**
**Touch on:**
- Testing across all 32 transformer layers
- 87.5% vulnerability rate
- This isn't layer-specific - it's architectural
- What this means for "conscious control"

**Key Information:**
- 28/32 layers vulnerable
- Only layers 16, 18, 22, 24 showed resistance
- Peak vulnerability: Layer 0 (40.71pp)
- Mid-layer resistance pattern

**Visuals:**
- Layer vulnerability heatmap
- Transformer architecture diagram showing vulnerable vs resistant layers
- Success rate by layer group chart

## **Section 5: Safety Implications**
**Touch on:**
- If elephant detection can be fooled...
- Real-world probe applications at risk
- Current AI safety systems vulnerability
- Defense challenges

**Key Information:**
- Bias detection probes in production systems
- Content moderation safety filters
- Code vulnerability detection
- Backdoor detection systems

**Visuals:**
- Threat model diagram: AI system → Probe → Safety decision
- Examples of vulnerable safety applications
- Attack surface visualization

## **Section 6: The Defense Challenge**
**Touch on:**
- Only 4 resistant layers found
- Why simple "robust layer" approach fails
- Need for ensemble methods
- Open research questions

**Key Information:**
- Resistant layers: 16, 18, 22, 24
- Ensemble defense concepts
- Multi-layer voting systems
- Defense-attack arms race

**Visuals:**
- Ensemble defense architecture diagram
- Vulnerability vs resistance comparison
- Defense strategy flowchart

## **Section 7: What This Means & What's Next**
**Touch on:**
- Implications for AI alignment
- Models have more control than expected
- Testing real safety-critical applications next
- Call for community engagement

**Key Information:**
- Next research directions (Blog Chunk 2)
- Broader research questions
- Community collaboration opportunities
- Responsible disclosure considerations

**Visuals:**
- Research roadmap timeline
- Community call-to-action graphic
- Future applications preview

## **Supporting Materials Needed:**
- Interactive demo (optional)
- Code repository link
- Experimental data availability
- Reproducibility instructions
- Contact information for collaboration

## **Technical Appendix (Optional):**
- Statistical significance details
- Experimental parameters
- Computational resources used
- Layer-by-layer detailed results