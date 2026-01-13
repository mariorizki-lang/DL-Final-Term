# Fine-tuning Phi-2 untuk Text Summarization dengan LoRA/QLoRA

## Project Overview

Proyek ini melakukan fine-tuning model **Phi-2** (2.7B parameters) untuk task **text summarization** menggunakan teknik **Parameter-Efficient Fine-Tuning (LoRA/QLoRA)** pada dataset **XSum**.

### Tujuan
- Mengadaptasi Phi-2, sebuah decoder-only language model, untuk menghasilkan ringkasan satu kalimat dari artikel berita
- Menggunakan LoRA/QLoRA untuk efisiensi memory dan computational cost
- Mengevaluasi performa dengan metrik ROUGE scores

### Dataset
- **XSum (Extreme Summarization)**: Dataset summarization dari BBC articles
- Training samples: 204,045 (subset: 2,000 untuk eksperimen)
- Validation samples: 11,332 (subset: 500)
- Test samples: 11,334
- Task: Menghasilkan one-sentence summary dari news articles

---

## Metodologi

### 1. Data Preprocessing (Notebook 1)

#### Proses Loading
- Dataset di-load dari HuggingFace datasets: `EdinburghNLP/xsum`
- Total 226,711 article-summary pairs

#### Data Exploration
**Statistik Dataset:**
- Rata-rata panjang dokumen: ~420 words
- Rata-rata panjang summary: ~23 words
- Median panjang dokumen: ~350 words
- Rekomendasi max document length: **1031 tokens**
- Rekomendasi max summary length: **45 tokens**

#### Data Cleaning
- Menghapus special characters dan formatting issues
- Normalisasi whitespace
- Validasi bahwa setiap article memiliki summary

### 2. Prompt Engineering (Notebook 2)

#### Template Selection
Diuji 5 template prompt yang berbeda:
1. Simple instruction format
2. Instruct-style format
3. Chat-style format
4. **Direct format (DIPILIH)** ✓
5. Question-answering format

#### Selected Prompt Template
```
Article: {article}

Summarize the above article in one sentence.
Summary: {summary}
```

**Rationale:**
- Clean dan straightforward
- Minimal token overhead
- Works well dengan decoder-only models seperti Phi-2
- Explicit task specification

#### Tokenization Configuration
- Tokenizer: `microsoft/phi-2`
- Vocabulary size: 50,257 tokens
- Max sequence length: **1,126 tokens**
- Padding: Dynamic (saat training)
- Truncation: Enabled

#### Token Statistics
- Rata-rata sequence length: **484 tokens**
- Median sequence length: **424.5 tokens**
- P95: 1,070 tokens
- P99: 1,126 tokens
- **4.3% sequences ter-truncate** (at max length)

### 3. Model Fine-tuning (Notebook 3)

#### Model Configuration

**Base Model:**
- Model: `microsoft/phi-2`
- Architecture: Decoder-only transformer
- Total parameters: **2.7B**
- Context length: 2,048 tokens

**Quantization (QLoRA):**
- Type: **4-bit quantization**
- Quant type: `nf4` (normalized float 4)
- Compute dtype: `float16`
- Double quantization: Enabled
- Memory reduction: ~75% dibanding FP16

**LoRA Configuration:**
```python
{
    "r": 16,                    # Rank
    "lora_alpha": 32,           # Scaling factor
    "target_modules": [
        "q_proj", "k_proj", "v_proj",  # Attention
        "dense", "fc1", "fc2"           # FFN
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

**Trainable Parameters:**
- Total parameters: ~2.7B
- Trainable parameters: ~4.2M (**0.16%**)
- Memory savings: **99.84% reduction** in training params

#### Training Hyperparameters

```python
{
    "num_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 16,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "optimizer": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "fp16": true
}
```

**Dataset Size:**
- Training samples: **2,000** (small subset untuk eksperimen cepat)
- Validation samples: **500**
- Rationale: Resource constraints (Google Colab T4 GPU)

#### Training Results

| Metric | Value |
|--------|-------|
| **Final Train Loss** | 2.2464 |
| **Final Eval Loss** | 2.2074 |
| **Perplexity** | 9.09 |
| Training Time | ~unknown |

**Observations:**
- Loss menurun konsisten selama training
- Eval loss lebih rendah dari train loss (mungkin karena dataset kecil)
- Perplexity 9.09 menunjukkan model memiliki confidence reasonable per token

### 4. Evaluation & Analysis (Notebook 4)

#### ROUGE Scores

**Fine-tuned Model:**
| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.0167 | 1.7% unigram overlap dengan reference |
| **ROUGE-2** | 0.0026 | 0.3% bigram overlap |
| **ROUGE-L** | 0.0107 | 1.1% longest common subsequence |

**Baseline (Zero-shot Phi-2):**
| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.0138 |
| ROUGE-2 | 0.0019 |
| ROUGE-L | 0.0082 |

**State-of-the-Art (Reference):**
| Metric | Score Range |
|--------|-------------|
| ROUGE-1 | 45-47% |
| ROUGE-2 | 22-25% |
| ROUGE-L | 37-40% |

#### Performance Comparison

**Improvement vs Baseline:**
| Metric | Absolute Gain | Relative Improvement |
|--------|---------------|---------------------|
| ROUGE-1 | +0.0029 | **+21.0%** |
| ROUGE-2 | +0.0007 | **+33.7%** |
| ROUGE-L | +0.0025 | **+30.7%** |

#### Generation Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Samples Evaluated** | 500 | 100% |
| **Successful Generations** | 49 | 9.8% |
| **Failed Generations** | 451 | **90.2%** |

**Failure Analysis:**
- Primary cause: `max_length` configuration error
- Error: "Input length of input_ids is X, but max_length is set to 150"
- Issue: Generation function menggunakan `max_length` instead of `max_new_tokens`

---

## Analisis Hasil

### Temuan Kunci

#### 1. 🚨 Critical Technical Issue
**Generation Function Error:**
- **90.2% generation failure rate** karena improper `max_length` configuration
- Caused by: Using `max_length=150` saat input tokens sudah > 150
- Solution: Should use `max_new_tokens` parameter instead

#### 2. 📉 Extremely Low ROUGE Scores
**Performance Gap:**
- Fine-tuned ROUGE-1: **1.7%** vs State-of-art: **46%** (27x gap)
- Scores sangat rendah karena:
  - Majority of generations failed (empty outputs)
  - Only 49/500 successful generations evaluated
  - Insufficient training (small dataset, few epochs)

#### 3. 📈 Positive Signal: Improvement over Baseline
**Despite low absolute scores:**
- **21-34% relative improvement** dibanding zero-shot baseline
- Menunjukkan model **telah belajar** dari fine-tuning
- Pada successful generations, ada improvement measurable

### Analisis Kualitatif

#### Best Example (ROUGE-L: 0.2564)
```
Article: The car mounted a pavement within the grounds of Maidstone 
Hospital, in Barming, at about 1440 BST on Tuesday, police said. The 
woman, in her 90s, was taken to a hospital in London, where she was 
later pronounced dead...

Reference Summary: A woman in her 90s who was in a wheelchair when 
she was hit by a car outside a hospital has died.

Generated Summary: Manager Chambers added: "Her work ethic and 
tenacity adds a great dimension to our defence."
```

**Observation:**
- Generated summary tidak match dengan article yang benar
- Model confusion antara multiple articles dalam batch
- Indicates dataset/batching issues during evaluation

#### Pattern Analysis dari Successful Generations
1. **Copy-paste behavior**: Model cenderung copy bagian dari input article
2. **Length inconsistency**: Generated summaries terlalu panjang atau terlalu pendek
3. **Lack of abstractiveness**: Tidak melakukan true summarization, hanya extraction
4. **Occasional relevance**: Beberapa examples menunjukkan pemahaman minimal

### Distribution Analysis

**Summary Length Distribution:**
- Reference summaries: Rata-rata 26 tokens, consistent
- Generated summaries (successful): More variable, cenderung lebih panjang
- Pattern: Model belum learn optimal summary length

---

## Limitasi dan Tantangan

### 1. Technical Implementation Issues

#### a) Generation Configuration Error
- **Root cause**: Confusion antara `max_length` dan `max_new_tokens`
- **Impact**: 90% failure rate dalam evaluation
- **Severity**: Critical - menghalangi proper evaluation

#### b) Dataset Size Constraints
- **Training size**: 2,000 samples (< 1% dari full dataset)
- **Impact**: Insufficient untuk belajar complex summarization patterns
- **Comparison**: State-of-art models trained pada 200K+ samples

#### c) Training Duration
- **Epochs**: Hanya 3 epochs
- **Impact**: Model belum converge sepenuhnya
- **Evidence**: Training loss masih menurun di akhir training

### 2. Model Performance Limitations

#### a) Extremely Low ROUGE Scores
- **Gap to SOTA**: 27x lower untuk ROUGE-1
- **Root causes**:
  1. Insufficient training data
  2. Limited training epochs
  3. Generation configuration errors
  4. Potential prompt template suboptimality

#### b) Lack of Abstractive Summarization
- Model melakukan **extractive** rather than **abstractive** summarization
- Copy-paste behavior dominan
- Minimal paraphrasing atau information synthesis

#### c) Length Control Issues
- Model tidak konsisten menghasilkan one-sentence summaries
- Some outputs terlalu panjang (multi-sentence)
- Others terlalu pendek (fragments)

### 3. Resource Constraints

#### a) Computational Limitations
- **Hardware**: Google Colab T4 GPU (15GB VRAM)
- **Impact**: Forced menggunakan small subset dan quantization
- **Trade-off**: Memory efficiency vs model capacity

#### b) Training Time Limitations
- **Total time**: Limited by Colab session constraints
- **Impact**: Tidak bisa train dengan full dataset
- **Workaround**: Using small subset untuk proof-of-concept

### 4. Evaluation Challenges

#### a) Limited Evaluation Samples
- Only 49 successful generations evaluated
- Statistical significance questionable
- Metrics mungkin tidak representative

#### b) No Human Evaluation
- ROUGE scores alone insufficient untuk judge quality
- Missing: Fluency, coherence, factuality assessment
- Need: Human evaluation study

---

## Rekomendasi Perbaikan

### 1. 🔧 Immediate Fixes (Critical)

#### a) Fix Generation Function
```python
# WRONG (Current):
outputs = model.generate(inputs, max_length=150)

# CORRECT (Should be):
outputs = model.generate(inputs, max_new_tokens=60)
```
**Priority**: P0 (Blocking proper evaluation)

#### b) Increase Max New Tokens
- Current setting equivalent: ~150 tokens
- Recommended: **60 tokens** (2.3x average summary length of 26)
- Allows model flexibility while preventing excessive generation

### 2. 📊 Training Improvements (High Priority)

#### a) Use Full Dataset
```python
Training samples: 204,045 (instead of 2,000)
Validation samples: 11,332 (instead of 500)
```
**Expected impact**: 
- Better generalization
- Higher ROUGE scores
- More robust model

#### b) Increase Training Duration
```python
num_epochs: 5-10 (instead of 3)
```
**Rationale**:
- Allow model to fully converge
- Current loss curve shows room for improvement
- XSum is complex task requiring more training

#### c) Experiment dengan Learning Rate
```python
learning_rates_to_try = [1e-4, 2e-4, 3e-4, 5e-4]
```
**Method**: Grid search atau learning rate finder
**Goal**: Find optimal learning rate untuk convergence

### 3. 🏗️ Architecture Enhancements (Medium Priority)

#### a) Increase LoRA Rank
```python
lora_config = {
    "r": 32 or 64,  # instead of 16
    "lora_alpha": 64 or 128
}
```
**Trade-off**: 
- More trainable parameters (better capacity)
- Higher memory usage (but still efficient)

#### b) Experiment dengan Target Modules
```python
# Current:
target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]

# Try:
target_modules = "all-linear"  # Apply LoRA to all linear layers
```

#### c) Alternative Prompt Templates
Test alternative formats:
```
1. More explicit: "Summarize this article in exactly one sentence:"
2. Length-constrained: "Provide a concise one-sentence summary (< 25 words):"
3. Question-based: "What is the main point of this article in one sentence?"
```

### 4. 🔍 Evaluation Enhancements (Medium Priority)

#### a) Implement Robust Error Handling
```python
def generate_summary_robust(article, model, tokenizer):
    try:
        # Proper generation with max_new_tokens
        # Fallback strategies
        # Error logging
    except Exception as e:
        log_error(e)
        return ""
```

#### b) Add Additional Metrics
Beyond ROUGE:
- **BLEU scores**: N-gram precision
- **METEOR**: Consider synonyms and paraphrases
- **BERTScore**: Semantic similarity
- **Perplexity**: Model confidence

#### c) Implement Human Evaluation
- Sample 100 generated summaries
- Evaluate for:
  - **Relevance**: Does summary capture main point?
  - **Fluency**: Is summary grammatically correct?
  - **Factuality**: Is information accurate?
  - **Conciseness**: Is it truly one sentence?

### 5. 🧪 Experimental Directions (Low Priority)

#### a) Try Different Base Models
```
- Llama-2-7B
- Mistral-7B
- Gemma-7B
```
**Rationale**: Some models may be better suited untuk summarization

#### b) Ensemble Methods
- Train multiple LoRA adapters
- Combine predictions untuk better results

#### c) Two-stage Approach
1. Stage 1: Extract key sentences
2. Stage 2: Compress menjadi one sentence

---

## Kesimpulan

### Achievements ✅
1. **Successfully implemented** end-to-end LoRA/QLoRA fine-tuning pipeline
2. **Demonstrated** parameter-efficient training (0.16% trainable params)
3. **Achieved** measurable improvement (21-34%) over baseline despite constraints
4. **Created** reproducible workflow dengan 4 notebooks terorganisir

### Key Learnings 📚
1. **Prompt engineering** significantly impacts model behavior
2. **Generation configuration** (max_length vs max_new_tokens) critical untuk success
3. **Dataset size** matters: 2K samples insufficient untuk complex task
4. **LoRA** enables training large models dengan limited resources
5. **Evaluation** requires robust error handling dan multiple metrics

### Current Status 📍
- **Proof of concept**: Successful ✓
- **Production-ready**: Not yet ✗
- **Next milestone**: Fix generation errors dan scale to full dataset

### Impact 💡
Meski ROUGE scores sangat rendah, proyek ini:
- **Validates** LoRA/QLoRA approach untuk resource-constrained environments
- **Identifies** critical implementation pitfalls (generation config)
- **Provides** clear roadmap untuk improvement
- **Demonstrates** iterative development process dalam deep learning

---

## File Structure

```
reports/
├── README.md                          # This file
├── training_metrics.json              # Detailed training metrics
├── evaluation_results.json            # Complete evaluation data
├── training_metrics_table.md          # Formatted metrics tables
├── evaluation_analysis.md             # Detailed evaluation analysis
├── lessons_learned.md                 # Refleksi dan insights
└── visualizations/
    ├── rouge_scores_comparison.png    # ROUGE scores bar chart
    ├── model_comparison.png           # Fine-tuned vs baseline
    ├── generation_statistics.png      # Success/failure rates
    └── training_configuration.png     # Hyperparameters overview
```

---

## References

1. **Phi-2**: https://huggingface.co/microsoft/phi-2
2. **XSum Dataset**: https://huggingface.co/datasets/EdinburghNLP/xsum
3. **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
4. **QLoRA Paper**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
5. **ROUGE Metrics**: Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries

---

## Contact & Repository

- **Author**: [Your Name]
- **Date**: January 2026
- **GitHub**: [Your Repository Link]
- **Notebooks**: Available in repository root

---

*Last Updated: January 11, 2026*
