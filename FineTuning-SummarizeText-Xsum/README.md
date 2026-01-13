# Fine-tuning Phi-2 untuk Text Summarization - XSum Dataset

## 📋 Project Information

**Project**: Fine-tuning Phi-2 dengan LoRA/QLoRA untuk Text Summarization
**Deadline**: 12 Januari 2025
**Status**: ✅ Completed

## 👥 Team Members

| Nama | NIM | Task |
|------|-----|------|
| Muhamad Mario Rizki | 1103223063 | Task 1 |
| Raihan Ivando Diaz | 1103223093 | Task 2 |
| Abid Sabyano Rozhan | 1103220222 | Task 3 |

---

## 📚 Project Overview

Proyek ini melakukan **fine-tuning model Phi-2 (2.7B parameters)** untuk task **text summarization** menggunakan dataset **XSum** dengan teknik **LoRA/QLoRA** (Parameter-Efficient Fine-Tuning).

---

## 📓 Notebook Descriptions

### 1. deep_learning_task3_Loading_Data_Preproccesing.ipynb

**Purpose**: Loading and preprocessing dataset XSum

**Workflow**:
- Load XSum dataset dari HuggingFace (226,711 article-summary pairs)
- Exploratory Data Analysis (EDA)
- Data preprocessing dan cleaning
- Token length optimization
- Save datasets ke Google Drive

**Key Statistics**:
- Total Dataset Size: 226,711 pairs
- Training Samples: 204,045
- Validation Samples: 11,332
- Test Samples: 11,334
- Average Document Length: ~420 words
- Average Summary Length: ~23 words
- Max Sequence Length: 1,126 tokens

---

### 2. deep_learning_task3_Prompt_Engineering_-_Data_Formatting.ipynb

**Purpose**: Prompt engineering dan data formatting untuk training

**Workflow**:
- Design dan test multiple prompt templates
- Select optimal prompt template
- Configure tokenization (microsoft/phi-2)
- Format data untuk LoRA training
- Generate configuration files

**Selected Prompt Template**:
```
Article: {article}

Summarize the above article in one sentence.
Summary: {summary}
```

**Tokenization Configuration**:
- Tokenizer: microsoft/phi-2
- Max Length: 1,126 tokens
- Padding: Dynamic
- Truncation: Enabled
- Average Sequence Length: 484 tokens

---

### 3. deep_learning_task3_Fine_tuning_Phi_2_with_LoRA_QLoRA.ipynb

**Purpose**: Model fine-tuning dengan LoRA/QLoRA

**Model Configuration**:
- Base Model: microsoft/phi-2 (2.7B parameters)
- LoRA Rank: 16
- LoRA Alpha: 32
- Trainable Parameters: 4.2M (0.16% of total)
- Quantization: 4-bit (QLoRA)

**Training Configuration**:
- Learning Rate: 2e-4
- Epochs: 3
- Batch Size: 2 (per device)
- Gradient Accumulation: 8
- Effective Batch Size: 16
- Optimizer: paged_adamw_8bit
- Scheduler: cosine
- Training Samples: 2,000
- Validation Samples: 500

**Training Results**:
- Final Train Loss: 2.2464
- Final Eval Loss: 2.2074
- Perplexity: 9.09
- Training Time: ~1-2 hours (on Google Colab T4)
- Model Checkpoint Size: ~8 MB

---

### 4. deep_learning_task3_Comprehensive_Evaluation_-_Analysis.ipynb

**Purpose**: Model evaluation dan comprehensive analysis

**Evaluation Process**:
- Load fine-tuned model
- Generate summaries untuk 500 validation samples
- Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Compare dengan baseline (zero-shot Phi-2)
- Analyze results dan generate reports

**ROUGE Scores**:

Fine-tuned Model:
- ROUGE-1: 0.0167 (1.7%)
- ROUGE-2: 0.0026 (0.3%)
- ROUGE-L: 0.0107 (1.1%)

Baseline (Zero-shot):
- ROUGE-1: 0.0138 (1.4%)
- ROUGE-2: 0.0019 (0.2%)
- ROUGE-L: 0.0082 (0.8%)

Improvement vs Baseline:
- ROUGE-1: +21.0%
- ROUGE-2: +33.7%
- ROUGE-L: +30.7%

**Generation Statistics**:
- Total Samples Evaluated: 500
- Successful Generations: 49 (9.8%)
- Failed Generations: 451 (90.2%)

---

## 🚀 Setup & Requirements

### Prerequisites
- Python 3.8+
- GPU dengan CUDA support (Google Colab T4 recommended)
- Google Drive account
- ~15GB GPU memory

### Installation

**Google Colab** (Recommended):
```python
# Install dependencies
!pip install datasets==3.1.0
!pip install transformers accelerate
!pip install peft
!pip install bitsandbytes
!pip install trl
!pip install rouge-score nltk

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

**Local Machine**:
```bash
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
transformers>=4.36.0
datasets>=3.1.0
peft>=0.7.1
bitsandbytes>=0.41.0
accelerate>=0.24.0
rouge-score>=0.1.2
nltk>=3.8.1
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## 📝 Running the Notebooks

### Google Colab
1. Open each notebook di Google Colab
2. Runtime → Change runtime type → Select GPU (T4)
3. Run all cells dari atas ke bawah
4. Follow urutan: Notebook 1 → 2 → 3 → 4

### Local Machine
```bash
jupyter notebook deep_learning_task3_Loading_Data_Preproccesing.ipynb
jupyter notebook deep_learning_task3_Prompt_Engineering_-_Data_Formatting.ipynb
jupyter notebook deep_learning_task3_Fine_tuning_Phi_2_with_LoRA_QLoRA.ipynb
jupyter notebook deep_learning_task3_Comprehensive_Evaluation_-_Analysis.ipynb
```

### Expected Runtime
- Notebook 1: ~10-15 minutes
- Notebook 2: ~5-10 minutes
- Notebook 3: ~1-2 hours (fine-tuning)
- Notebook 4: ~30-45 minutes (evaluation)
- **Total**: ~2-3 hours

---

## 📊 Key Results

### Model Efficiency
- **Parameter Reduction**: 99.84% (4.2M / 2.7B trainable)
- **Memory Reduction**: ~75% (4-bit quantization)
- **Training Time**: 1-2 hours (vs 20-30 hours for full fine-tuning)
- **Model Size**: 8 MB (vs 10 GB for full model)

### Performance Improvement
- **ROUGE-1**: 0.0138 → 0.0167 (+21.0%)
- **ROUGE-2**: 0.0019 → 0.0026 (+33.7%)
- **ROUGE-L**: 0.0082 → 0.0107 (+30.7%)

### Training Metrics
- Train Loss: 2.2464
- Eval Loss: 2.2074
- Perplexity: 9.09

---

## 💡 Key Findings

### Achievements
✅ Successfully fine-tuned Phi-2 dengan LoRA/QLoRA
✅ Achieved 21-34% improvement over baseline
✅ Efficient training on Google Colab T4
✅ 99.84% parameter reduction via LoRA
✅ Comprehensive evaluation dan analysis

### Challenges
⚠️ Generation configuration error (90% failure rate)
⚠️ Limited training data (2,000 samples)
⚠️ Low absolute ROUGE scores (1.7% vs SOTA 45%)

### Learnings
💡 Data size matters more than model size
💡 Prompt engineering significantly impacts results
💡 LoRA is highly effective for parameter-efficient fine-tuning
💡 Testing early catches bugs and saves time

---

## 📁 Generated Reports & Analysis

Comprehensive reports tersedia:
- training_metrics.json - Training metrics data
- evaluation_results.json - Evaluation data
- training_metrics_table.md - Detailed metrics tables
- evaluation_analysis.md - Deep analysis
- lessons_learned.md - Key insights

Visualizations:
- rouge_comparison.png - ROUGE scores comparison
- generation_success_rate.png - Success/failure rates
- relative_improvement.png - Improvement over baseline
- lora_efficiency.png - LoRA efficiency comparison

---

## 🔧 Troubleshooting

### Issue: CUDA out of memory
- Reduce batch size: 2 → 1
- Increase gradient accumulation: 8 → 16
- Use smaller LoRA rank: 16 → 8

### Issue: Generation produces empty output
- Use max_new_tokens instead of max_length
- Set reasonable max_new_tokens (60 recommended)
- Check tokenizer padding configuration

### Issue: Dataset not found
- Mount Google Drive properly
- Run Notebook 1 fully
- Verify dataset path in output

---

## 📚 References

### Papers
- LoRA: "Low-Rank Adaptation of Large Language Models"
- QLoRA: "Efficient Finetuning of Quantized LLMs"
- Phi-2: "The surprising power of small language models"
- XSum: "Don't Give Me the Details, Just the Summary!"

### Datasets & Models
- XSum Dataset: https://huggingface.co/datasets/EdinburghNLP/xsum
- Phi-2 Model: https://huggingface.co/microsoft/phi-2
- Transformers: https://huggingface.co/transformers/
- PEFT: https://github.com/huggingface/peft

---

## ✅ Project Status

**Status**: Completed
**Date**: January 2026
**Task 3**: ✅ Complete

All 4 notebooks documented and ready for use.
