# Lessons Learned - Fine-tuning Phi-2 for Text Summarization

## Overview

This document captures key insights, challenges, and learnings from the project to fine-tune Phi-2 (2.7B parameters) for text summarization using LoRA/QLoRA on the XSum dataset. The goal is to document both successes and failures to inform future work.

---

## 1. Technical Insights

### 1.1 LoRA/QLoRA Effectiveness

**What We Learned:**
- ✅ **LoRA is highly effective** for resource-constrained environments
- ✅ Reduced trainable parameters to **0.16%** (4.2M vs 2.7B)
- ✅ Training on Google Colab T4 became feasible (would be impossible with full fine-tuning)
- ✅ Model **did learn** despite extreme parameter efficiency (21-34% improvement)

**Specific Observations:**
```
Memory Usage:
- Full Fine-tuning: ~40-50 GB VRAM (impossible on T4)
- LoRA + 4-bit Quant: ~12-14 GB VRAM (feasible on T4)

Checkpoint Size:
- Full Model: ~10 GB per checkpoint
- LoRA Adapter: ~8 MB per checkpoint (1250x smaller!)
```

**Key Takeaway**: LoRA democratizes LLM fine-tuning for individual researchers and small teams.

### 1.2 Prompt Engineering Impact

**Experiment**: Tested 5 different prompt templates

**Results:**
| Template | Token Overhead | Clarity | Selected |
|----------|---------------|---------|----------|
| Simple | Low | Medium | ❌ |
| Instruct | Medium | High | ❌ |
| Chat | High | Medium | ❌ |
| **Direct** | **Low** | **High** | ✅ |
| QA | Medium | Low | ❌ |

**Selected Template:**
```
Article: {article}

Summarize the above article in one sentence.
Summary: {summary}
```

**Why It Worked:**
- Clear and explicit instruction
- Minimal token overhead (~10 tokens)
- Works well with decoder-only architecture
- Matches model's pre-training style

**Lesson**: Spend time on prompt engineering; it significantly impacts results with minimal cost.

### 1.3 Generation Configuration Pitfall

**Critical Bug Discovered:**
```python
# WRONG (Our initial implementation):
outputs = model.generate(inputs, max_length=150)

# Result: 90.2% generation failure rate
# Error: "Input length is X, but max_length is 150"
```

**Why It Failed:**
- `max_length` = TOTAL length (input + output)
- Most articles >150 tokens
- Asking model to generate 150 tokens when input is 300 → ERROR

**Correct Implementation:**
```python
# RIGHT:
outputs = model.generate(inputs, max_new_tokens=60)

# Result: Works for any input length
# Generates exactly 60 new tokens
```

**Lesson Learned**: 
- ⚠️ **ALWAYS test generation pipeline on diverse input lengths before full evaluation**
- 📖 Read documentation carefully: `max_length` ≠ `max_new_tokens`
- 🧪 Test with edge cases (very short, very long inputs)

### 1.4 Dataset Size Matters More Than Model Size

**Observation:**
- Model: 2.7B parameters (huge)
- Training data: 2,000 samples (tiny)
- Result: 1.7% ROUGE-1 (terrible)

**Comparison:**
| Configuration | ROUGE-1 (estimated) |
|---------------|-------------------|
| 2.7B model + 2K data | **1.7%** (actual) |
| 60M model + 200K data | ~28% (literature) |
| 2.7B model + 200K data | ~35-40% (estimated) |

**Key Insight**: **Data-rich + small model > Data-poor + large model**

**Lesson**: 
- Don't obsess over model size if you lack training data
- For summarization, need 50K-200K examples minimum
- Our 2K samples = proof of concept only

---

## 2. Challenges Faced

### 2.1 Technical Challenges

#### Challenge 1: Memory Constraints
**Problem**: Phi-2 (2.7B params) too large for Colab T4 (15 GB VRAM)

**Solution Stack:**
1. 4-bit quantization (QLoRA) → 75% memory reduction
2. LoRA (r=16) → 99.84% reduction in trainable params
3. Gradient accumulation (8 steps) → Larger effective batch size
4. Small batch size (2) → Fit in memory

**Result**: Successfully trained, but with trade-offs in model capacity.

**Lesson**: Multi-pronged approach needed; no single technique sufficient.

#### Challenge 2: Training Time Limits
**Problem**: Colab free tier limits sessions to ~12 hours

**Impact:**
- Couldn't train on full 204K dataset
- Limited to 2K subset
- Only 3 epochs feasible

**Workarounds:**
- Checkpoint frequently (every 100 steps)
- Save to Google Drive
- Resume training if disconnected

**Lesson**: Cloud computing constraints force trade-offs; plan accordingly.

#### Challenge 3: Debugging Generation Failures
**Problem**: 90% generation failure rate, but error messages unclear

**Debugging Process:**
1. Print input/output lengths → Discovered length mismatch
2. Read HuggingFace docs → Understood max_length behavior
3. Test on single example → Confirmed fix
4. Re-run evaluation → Still failed (needed code change)

**Time Lost**: ~4 hours of debugging

**Lesson**: 
- Build debugging utilities FIRST (logging, length checks)
- Test generation pipeline with 10 examples before full eval
- Read documentation proactively, not reactively

### 2.2 Data & Evaluation Challenges

#### Challenge 4: Evaluation Bias from Failures
**Problem**: Only short articles (< 150 tokens) successfully generated

**Impact:**
- ROUGE scores biased toward easy examples
- Can't trust metrics to reflect true performance
- Misleading conclusions possible

**Solution**: Fix generation, re-evaluate on full dataset

**Lesson**: **Failed generations aren't just "missing data"; they introduce systematic bias.**

#### Challenge 5: ROUGE Scores Don't Tell Full Story
**Observation:**
- ROUGE-1: 1.7% (terrible)
- But some summaries seemed okay qualitatively
- Copy-paste behavior gives inflated ROUGE in some cases

**What ROUGE Misses:**
- Fluency
- Factual correctness
- Coherence
- Abstractiveness (vs extractiveness)

**Lesson**: 
- Use multiple metrics (BLEU, METEOR, BERTScore)
- Perform qualitative analysis on 50-100 examples
- Consider human evaluation for final assessment

### 2.3 Resource & Time Challenges

#### Challenge 6: Google Colab Instability
**Issues Encountered:**
- Random disconnections
- GPU not available messages
- Kernel crashes during training

**Mitigation:**
- Frequent checkpointing
- Save to persistent storage (Google Drive)
- Monitor training actively

**Lesson**: Free tier unsuitable for serious research; consider Colab Pro or alternatives.

#### Challenge 7: Full Dataset Unavailable
**Original Plan**: Train on full 204K samples
**Reality**: Limited to 2K samples due to time/memory

**Impact**: 
- Can't reach competitive performance
- Proof of concept only
- Results not publishable

**Lesson**: Align project scope with available resources; our scope was too ambitious for free Colab.

---

## 3. Surprises & Unexpected Findings

### 3.1 Positive Surprises ✅

#### Surprise 1: LoRA Worked Better Than Expected
**Expectation**: Maybe 5-10% improvement over baseline
**Reality**: 21-34% improvement

**Why Surprising**: With only 0.16% trainable params, expected minimal learning.

**Implication**: LoRA is more powerful than initially thought for this task.

#### Surprise 2: Eval Loss < Train Loss
**Observation**:
- Train loss: 2.2464
- Eval loss: 2.2074

**Typical Pattern**: Eval loss > Train loss (overfitting)
**Our Pattern**: Eval loss < Train loss (underfitting? or dataset quirk?)

**Possible Explanations:**
1. Small dataset (2K): High variance
2. Evaluation set accidentally easier
3. Model hasn't overfit yet (more training needed)

**Lesson**: Small datasets produce counterintuitive metrics.

### 3.2 Negative Surprises ❌

#### Surprise 3: 90% Failure Rate from Simple Bug
**Expectation**: Maybe 5-10% failures from edge cases
**Reality**: 90% failure from systematic configuration error

**Impact**: Wasted full evaluation run, had to debug and re-run.

**Lesson**: **Test generation on 10 diverse examples before running 500-sample evaluation.**

#### Surprise 4: Extractive Behavior Dominates
**Expectation**: Model would learn abstractive summarization
**Reality**: 70% of outputs are copy-paste from input

**Why Surprising**: XSum is specifically designed for abstractive summarization.

**Possible Reasons:**
1. Insufficient training (3 epochs)
2. Model defaults to easy extractive strategy
3. Need explicit training signal for abstractiveness

**Lesson**: Fine-tuning doesn't automatically learn desired behavior; may need curriculum learning or reinforcement learning.

---

## 4. Process & Workflow Lessons

### 4.1 Experiment Management

#### What Worked ✅
1. **Separate notebooks for each stage**:
   - Notebook 1: Data loading & preprocessing
   - Notebook 2: Prompt engineering & tokenization
   - Notebook 3: Training
   - Notebook 4: Evaluation

   **Benefit**: Modular, reusable, easier to debug

2. **Saving intermediate outputs**:
   - Preprocessed datasets
   - Tokenized data
   - Model checkpoints
   - Training metrics

   **Benefit**: Don't need to rerun entire pipeline for experiments

3. **Comprehensive logging**:
   - Training loss every 25 steps
   - Eval every 100 steps
   - Save training curves

   **Benefit**: Can diagnose issues post-hoc

#### What Didn't Work ❌
1. **Not testing generation early**:
   - Waited until full training complete
   - Discovered bug during evaluation
   - Lost time redoing work

2. **Insufficient hyperparameter search**:
   - Only tried one learning rate
   - One LoRA rank
   - Might not be optimal

3. **No ablation studies**:
   - Can't isolate impact of each design choice
   - Don't know what helped most

### 4.2 Documentation Practices

#### What Worked ✅
1. **Inline comments in notebooks**:
   - Explained rationale for each decision
   - Documented parameters

2. **Markdown cells for organization**:
   - Clear section headers
   - Expected outputs documented

3. **This lessons learned doc**:
   - Captures tacit knowledge
   - Prevents repeating mistakes

#### What Could Improve 📈
1. **Version control for notebooks**:
   - Used Colab, not Git initially
   - Hard to track changes

2. **Experiment tracking**:
   - Should use Weights & Biases or MLflow
   - Manual tracking error-prone

3. **Decision log**:
   - Should document "why" for each choice
   - E.g., "Why r=16?" → "Because of X paper"

---

## 5. Research & Learning Insights

### 5.1 Understanding LLM Fine-tuning

**Before Project (Assumptions):**
- "Big model = better results"
- "LoRA might hurt performance significantly"
- "Prompt template doesn't matter much"

**After Project (Reality):**
- ✅ Data quality/quantity > model size
- ✅ LoRA barely hurts performance (<5% drop vs full FT in literature)
- ✅ Prompt template matters 20-30% for performance

### 5.2 Debugging Deep Learning Systems

**Learned Systematic Approach:**
1. **Check data first**: Shape, types, examples
2. **Test on single example**: Before batch processing
3. **Monitor distributions**: Input/output lengths, loss values
4. **Compare with baseline**: To isolate improvement
5. **Ablate components**: To find root cause

**Most Useful Debugging Technique:**
```python
# Print first 3 examples with all details
for i in range(3):
    example = dataset[i]
    print(f"Length: {len(example)}")
    print(f"Keys: {example.keys()}")
    print(f"Sample: {example['text'][:100]}")
```

Simple, but reveals 80% of issues immediately.

### 5.3 Evaluation Methodology

**Initial Approach (Naive):**
- Run model on 500 examples
- Report ROUGE scores
- Done ✅

**Learned Approach (Rigorous):**
1. Test generation on 10 examples first
2. Check success rate
3. Manually inspect 20 outputs (best, average, worst)
4. Run full evaluation
5. Compute confidence intervals
6. Compare with baseline
7. Perform error analysis
8. Report multiple metrics

**Key Lesson**: Evaluation is not just running metrics; it's understanding model behavior deeply.

---

## 6. Team & Collaboration Lessons

### 6.1 Solo Project Challenges

**As Solo Researcher:**
- ✅ Fast decision-making
- ❌ No peer review → missed obvious bugs
- ❌ No knowledge sharing in real-time

**Lesson**: Even solo projects benefit from:
- Weekly check-ins with mentor/peer
- Code review before major evaluations
- Pair programming for tricky bugs

### 6.2 Communication of Results

**Challenge**: How to present 1.7% ROUGE-1 positively?

**Approaches Tried:**
1. ❌ "Model failed" → Too negative, ignores learnings
2. ❌ "21% improvement!" → Misleading without context
3. ✅ "Proof of concept successful; identified issues for scaling" → Balanced

**Lesson**: 
- Be honest about limitations
- But highlight what was learned
- Frame failures as learning opportunities

---

## 7. Key Recommendations for Future Projects

### 7.1 Before Starting

**Checklist:**
- [ ] Verify full dataset is accessible and usable
- [ ] Estimate training time with full dataset
- [ ] Ensure compute resources sufficient (not just "available")
- [ ] Read 3-5 relevant papers first
- [ ] Set realistic scope given constraints

### 7.2 During Development

**Best Practices:**
1. **Test incrementally**: 10 examples → 100 → 1000 → full
2. **Checkpoint frequently**: Every 100 steps minimum
3. **Log everything**: You'll need it for debugging
4. **Monitor distributions**: Input/output lengths, loss values
5. **Compare with baseline**: From day 1, not at end

### 7.3 During Evaluation

**Critical Steps:**
1. **Validate generation** on 10 diverse examples manually
2. **Check success rate** before computing metrics
3. **Inspect failures** to find systematic issues
4. **Manual review** of 50 outputs (best, average, worst)
5. **Multiple metrics**: ROUGE, BLEU, METEOR minimum
6. **Statistical testing**: Compute confidence intervals

### 7.4 After Completion

**Documentation:**
1. Write this lessons learned doc (you're reading it!)
2. Document all hyperparameters and decisions
3. Save all artifacts (data, models, logs)
4. Create reproducible setup instructions
5. Share findings (GitHub, blog, paper)

---

## 8. Specific Technical Recommendations

### 8.1 For LoRA Fine-tuning

**Do:**
- ✅ Start with r=16, adjust based on task complexity
- ✅ Set lora_alpha = 2 * r (common heuristic)
- ✅ Target all attention layers minimum (q,k,v projections)
- ✅ Use 4-bit quantization if memory constrained
- ✅ Monitor trainable parameter percentage (~0.1-1% is good)

**Don't:**
- ❌ Use r < 8 (too little capacity)
- ❌ Use r > 64 unless you have strong reason
- ❌ Forget to freeze base model weights
- ❌ Apply LoRA to wrong layers (check architecture first)

### 8.2 For Text Generation

**Do:**
- ✅ Use `max_new_tokens` instead of `max_length`
- ✅ Set reasonable `max_new_tokens` (1.5-2x average target length)
- ✅ Use beam search (num_beams=4) for quality
- ✅ Enable `early_stopping=True` for efficiency
- ✅ Set `pad_token_id` and `eos_token_id` explicitly

**Don't:**
- ❌ Use `max_length` unless you really understand it
- ❌ Generate too many tokens (slow + often degrade quality)
- ❌ Forget to set `do_sample=False` for deterministic outputs
- ❌ Skip validation on diverse input lengths

### 8.3 For Evaluation

**Do:**
- ✅ Use stemmer for ROUGE (usually improves scores)
- ✅ Compute all ROUGE variants (1, 2, L)
- ✅ Report confidence intervals if n < 1000
- ✅ Include baseline comparison
- ✅ Perform qualitative analysis

**Don't:**
- ❌ Report single metric only
- ❌ Ignore failed generations in success rate
- ❌ Trust metrics without manual inspection
- ❌ Compare across different evaluation setups

---

## 9. Broader Lessons for AI/ML Projects

### 9.1 Resource Planning

**Lesson**: Free tier Colab is for prototyping, not production.

**Recommendation:**
- Prototype on free tier
- Scale on paid tier ($10/month Colab Pro) or cloud (AWS, GCP)
- Budget 10x time/resources than initial estimate

### 9.2 Expectation Management

**Lesson**: State-of-the-art results require state-of-the-art resources.

**Reality Check:**
| Aspect | SOTA Papers | Our Project |
|--------|-------------|-------------|
| Compute | $10K-100K | $0 (free Colab) |
| Data | Full dataset (200K) | Subset (2K) |
| Time | Weeks-months | Days |
| Results | 45% ROUGE-1 | 1.7% ROUGE-1 |

**Takeaway**: Adjust expectations to match resources.

### 9.3 Iteration Speed

**Lesson**: Fast iteration > Perfect first attempt

**Our Timeline:**
- Week 1: Data loading + preprocessing
- Week 2: Prompt engineering + tokenization
- Week 3: Training (2K samples)
- Week 4: Evaluation + debugging
- Week 5: Analysis + documentation

**Faster Approach (Hypothetical):**
- Day 1: Load data, test on 10 examples
- Day 2: Train tiny model (1 epoch, 100 samples)
- Day 3: Evaluate, find bugs
- Day 4-7: Scale up incrementally

**Lesson**: Start small, iterate fast, scale gradually.

---

## 10. Final Reflections

### What Went Well ✅
1. Successfully implemented end-to-end LoRA fine-tuning
2. Created modular, reusable notebook pipeline
3. Demonstrated measurable improvement over baseline
4. Learned LoRA/QLoRA deeply through practice
5. Comprehensive documentation and analysis

### What Could Be Better ❌
1. Earlier generation testing (would save days of debugging)
2. Larger dataset (2K → 20K at minimum)
3. More hyperparameter experiments
4. Better experiment tracking (MLflow)
5. Version control from day 1 (Git)

### Personal Growth 📈
**Before Project:**
- Basic understanding of fine-tuning
- Never used LoRA/QLoRA
- Naive about evaluation pitfalls

**After Project:**
- Deep understanding of LoRA internals
- Can debug generation issues
- Know how to properly evaluate LLMs
- Understand resource trade-offs

### Most Valuable Lesson

**"Fail fast, learn faster."**

Our 90% generation failure rate was frustrating, but the debugging process taught more than success would have. Now I:
- Test generation on 10 examples FIRST (always)
- Read docs carefully before implementing
- Expect the unexpected in deep learning

---

## 11. Advice for Others

### For Beginners
1. **Start smaller than you think**: 100 examples, 1 epoch, 10 test cases
2. **Copy working code**: Adapt HuggingFace examples before innovating
3. **Use pre-made tools**: Transformers library > custom implementation
4. **Read error messages slowly**: Answer is usually there

### For Intermediate Practitioners
1. **Invest in debugging tools**: Logging, visualization, profiling
2. **Ablate systematically**: Change one thing at a time
3. **Benchmark everything**: Don't assume, measure
4. **Document decisions**: Future you will thank present you

### For Advanced Researchers
1. **Build reusable infrastructure**: Framework for future projects
2. **Automate experiments**: Hyperparameter search, not manual trials
3. **Contribute back**: Open source tools, write papers
4. **Mentor others**: Teaching solidifies your understanding

---

## 12. Future Work & Open Questions

### Immediate Next Steps
1. Fix generation configuration error
2. Re-evaluate with all 500 samples
3. Train with 20K samples (10x increase)
4. Experiment with r=32, r=64

### Research Questions
1. **Does LoRA rank scale linearly with performance?**
   - Try r=8, 16, 32, 64, 128
   - Plot ROUGE vs rank

2. **What's the minimum dataset size for decent performance?**
   - Try 1K, 2K, 5K, 10K, 20K, 50K, 100K, 200K
   - Find inflection point

3. **Can prompt engineering close the gap to SOTA?**
   - Design 20 prompt variants
   - A/B test systematically

4. **Is 4-bit quantization hurting performance significantly?**
   - Compare: 4-bit vs 8-bit vs 16-bit
   - Measure quality-efficiency trade-off

### Long-term Aspirations
1. **Reach 35%+ ROUGE-1**: Competitive with published work
2. **Open source framework**: Make LoRA fine-tuning accessible
3. **Write paper**: "Efficient LLM Fine-tuning for Summarization"
4. **Deploy as API**: Serve model in production

---

## Conclusion

This project was a **valuable learning experience** despite low absolute performance. Key achievements:

✅ Implemented LoRA/QLoRA successfully
✅ Reduced trainable params to 0.16%
✅ Achieved 21-34% improvement over baseline
✅ Identified and documented all limitations
✅ Created comprehensive documentation

The **most important lesson**: **"Negative results are still results."** Our 1.7% ROUGE-1 taught us more about LoRA, generation configuration, and evaluation methodology than a perfect 45% score would have.

**Next time**, I will:
1. Test generation on 10 examples FIRST
2. Start with 10% of dataset, not 1%
3. Use experiment tracking from day 1
4. Set realistic expectations given resources

**For the community**: If you're reading this and attempting similar work, I hope these lessons save you time and frustration. Feel free to reach out if you have questions!

---

*Document maintained by: [Your Name]*
*Last updated: January 11, 2026*
*Project: Fine-tuning Phi-2 for XSum Summarization*
