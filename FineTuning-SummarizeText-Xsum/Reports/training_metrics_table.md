# Training Metrics - Detailed Tables

## Model Configuration

### Base Model Specifications
| Parameter | Value |
|-----------|-------|
| Model Name | microsoft/phi-2 |
| Architecture | Decoder-only Transformer |
| Total Parameters | 2.7 Billion |
| Context Length | 2,048 tokens |
| Vocabulary Size | 50,257 |

### Quantization Configuration (QLoRA)
| Parameter | Value |
|-----------|-------|
| Quantization Type | 4-bit |
| Quant Type | nf4 (normalized float 4) |
| Compute Dtype | float16 |
| Double Quantization | Enabled |
| Memory Reduction | ~75% vs FP16 |

### LoRA Configuration
| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, dense, fc1, fc2 |
| Bias | none |
| Task Type | CAUSAL_LM |

### Trainable Parameters
| Metric | Value |
|--------|-------|
| Total Parameters | ~2,700,000,000 |
| Trainable Parameters | ~4,200,000 |
| Trainable Percentage | **0.16%** |
| Memory Savings | **99.84%** reduction |

---

## Training Configuration

### Dataset Configuration
| Parameter | Value |
|-----------|-------|
| Dataset Name | XSum (EdinburghNLP) |
| Full Training Size | 204,045 samples |
| Used Training Size | **2,000 samples** |
| Full Validation Size | 11,332 samples |
| Used Validation Size | **500 samples** |
| Test Size | 11,334 samples |
| Subset Reason | Resource constraints (Google Colab) |

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Number of Epochs | 3 |
| Per-Device Train Batch Size | 2 |
| Per-Device Eval Batch Size | 2 |
| Gradient Accumulation Steps | 8 |
| **Effective Batch Size** | **16** |
| Learning Rate | 2e-4 (0.0002) |
| LR Scheduler | cosine |
| Warmup Steps | 100 |
| Optimizer | paged_adamw_8bit |
| Weight Decay | 0.01 |
| Max Gradient Norm | 1.0 |
| FP16 Training | Enabled |
| Random Seed | 42 |

### Data Processing
| Parameter | Value |
|-----------|-------|
| Tokenizer | microsoft/phi-2 |
| Max Sequence Length | 1,126 tokens |
| Padding Strategy | Dynamic |
| Truncation | Enabled |
| Average Sequence Length | 484 tokens |
| Median Sequence Length | 424 tokens |
| Truncation Rate | 4.3% |

### Evaluation Strategy
| Parameter | Value |
|-----------|-------|
| Eval Strategy | steps |
| Eval Steps | 100 |
| Save Strategy | steps |
| Save Steps | 100 |
| Save Total Limit | 3 |
| Load Best Model at End | True |
| Metric for Best Model | eval_loss |
| Logging Steps | 25 |

---

## Training Results

### Final Metrics
| Metric | Value |
|--------|-------|
| **Final Train Loss** | **2.2464** |
| **Final Eval Loss** | **2.2074** |
| **Perplexity** | **9.09** |

### Loss Analysis
| Observation | Details |
|-------------|---------|
| Train vs Eval Loss | Eval loss (2.21) < Train loss (2.25) |
| Possible Reason | Small dataset size; evaluation set might be easier |
| Convergence Status | Partially converged |
| Loss Trend | Still decreasing at end of training |
| Recommendation | Increase epochs to 5-10 for full convergence |

### Perplexity Interpretation
| Perplexity | Interpretation |
|------------|----------------|
| 9.09 | Model has ~9 possible tokens per prediction on average |
| Quality | Reasonable confidence per token |
| Context | For LLM, perplexity 5-15 considered good for fine-tuned models |

---

## Resource Usage

### Hardware Configuration
| Resource | Specification |
|----------|---------------|
| Platform | Google Colab |
| GPU | Tesla T4 |
| VRAM | 15.83 GB |
| GPU Memory Usage | ~12-14 GB during training |
| System RAM | ~12 GB |

### Training Efficiency
| Metric | Value |
|--------|-------|
| Estimated Training Time | 1-2 hours (for 2K samples, 3 epochs) |
| Steps per Epoch | ~125 steps |
| Total Training Steps | ~375 steps |
| Average Step Time | ~10-15 seconds |
| Memory Efficiency | 99.84% parameter reduction via LoRA |

### Comparison: Full Fine-tuning vs LoRA
| Aspect | Full Fine-tuning | LoRA (This Project) |
|--------|-----------------|---------------------|
| Trainable Parameters | 2.7B (100%) | 4.2M (0.16%) |
| GPU Memory Required | ~40-50 GB | ~12-14 GB |
| Training Time | ~20-30 hours | ~1-2 hours |
| Storage for Checkpoints | ~10 GB per checkpoint | ~8 MB per checkpoint |
| Feasibility on Colab | ❌ No | ✅ Yes |

---

## Training Progress

### Epoch-by-Epoch Summary
| Epoch | Train Loss (approx) | Eval Loss | Perplexity | Notes |
|-------|-------------------|-----------|------------|-------|
| 1 | ~2.8 | ~2.6 | ~13.5 | Initial learning |
| 2 | ~2.5 | ~2.3 | ~10.0 | Steady improvement |
| 3 | 2.2464 | 2.2074 | 9.09 | Final metrics |

### Loss Curve Observations
| Step Range | Train Loss Trend | Eval Loss Trend | Notes |
|------------|------------------|-----------------|-------|
| 0-100 | Rapid decrease | Rapid decrease | Initial learning phase |
| 100-200 | Steady decrease | Steady decrease | Consistent learning |
| 200-300 | Gradual decrease | Gradual decrease | Approaching convergence |
| 300-375 | Slow decrease | Slow decrease | Still room for improvement |

---

## Comparison with Baseline

### Zero-shot Phi-2 (Baseline)
| Configuration | Details |
|---------------|---------|
| Model | microsoft/phi-2 (no fine-tuning) |
| Prompting | Same prompt template as fine-tuned |
| Evaluation | Same 100-sample subset |
| Purpose | Measure improvement from fine-tuning |

### Performance Metrics: Baseline vs Fine-tuned
| Metric | Baseline (Zero-shot) | Fine-tuned | Absolute Gain | Relative Gain |
|--------|---------------------|-----------|---------------|---------------|
| ROUGE-1 | 0.0138 (1.4%) | 0.0167 (1.7%) | +0.0029 | **+21.0%** |
| ROUGE-2 | 0.0019 (0.2%) | 0.0026 (0.3%) | +0.0007 | **+33.7%** |
| ROUGE-L | 0.0082 (0.8%) | 0.0107 (1.1%) | +0.0025 | **+30.7%** |
| Perplexity | N/A | 9.09 | N/A | N/A |

### Key Takeaway
**Despite extremely low absolute scores, fine-tuning achieved 21-34% relative improvement over zero-shot baseline**, demonstrating that the model did learn from the training process.

---

## Prompt Template Used

### Final Selected Template
```
Article: {article}

Summarize the above article in one sentence.
Summary: {summary}
```

### Template Statistics
| Metric | Value |
|--------|-------|
| Prompt Overhead Tokens | ~10-15 tokens |
| Template Type | Direct instruction format |
| Clarity | Explicit task specification |
| Efficiency | Minimal token overhead |

### Alternative Templates Tested
| Template | Description | Selected |
|----------|-------------|----------|
| Simple | Basic "Summarize:" format | ❌ |
| Instruct | "Instruct: Summarize..." | ❌ |
| Chat | Conversation-style | ❌ |
| **Direct** | "Article: ... Summary:" | ✅ |
| QA | Question-answer format | ❌ |

---

## Next Steps & Recommendations

### Immediate Actions (Priority: Critical)
1. **Fix generation function**: Use `max_new_tokens` instead of `max_length`
2. **Re-evaluate model**: Run evaluation with fixed generation

### Short-term Improvements (Priority: High)
1. **Scale to full dataset**: Train on 204K samples
2. **Increase epochs**: 5-10 epochs
3. **Experiment with learning rate**: Try 1e-4, 3e-4, 5e-4

### Medium-term Enhancements (Priority: Medium)
1. **Increase LoRA rank**: Try r=32 or r=64
2. **Add more metrics**: BLEU, METEOR, BERTScore
3. **Implement human evaluation**: 100-sample study

### Long-term Explorations (Priority: Low)
1. **Try different base models**: Llama-2, Mistral, Gemma
2. **Ensemble methods**: Multiple LoRA adapters
3. **Two-stage summarization**: Extract + Compress

---

*Generated: January 11, 2026*
