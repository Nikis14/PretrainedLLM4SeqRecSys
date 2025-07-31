# Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation

This repository contains code for the paper ["Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation"]()

## Abstract

Sequential recommender systems have achieved significant success in modeling temporal user behavior, but remain limited in capturing rich user semantics beyond basic interaction patterns. Large Language Models (LLMs) present opportunities to enhance user understanding with their reasoning capabilities, yet existing integration approaches create prohibitive inference costs in real time. To address these limitations, we present a novel knowledge distillation method that transfers user-centric knowledge from pre-trained LLMs into sequential recommenders without requiring LLM inference at serving time. Our approach generates rich user profiles using LLM, then teaches sequential models to reconstruct these profiles through an auxiliary loss, effectively embedding semantic reasoning into model parameters. This design maintains the inference efficiency of traditional sequential models while requiring neither architectural modifications nor LLM fine-tuning. Extensive evaluation across four diverse datasets demonstrates substantial improvements in recommendation quality. Notably, our method achives up to 72% improvement in cold-start scenario for items — addressing a critical challenge in recommender systems. This work demonstrates a practical pathway for efficient incorporation of pre-trained LLM capabilities into recommender systems.

## Method Overview

Our approach consists of two main components: **LLM-based User Profile Generation** and **Knowledge Distillation to Sequential Recommender Models**. The key innovation is decoupling the recommendation process from LLM inference, making the system production-ready while retaining the benefits of LLM-derived semantic understanding.

### Problem Formulation

Let $\mathcal{U}$ and $\mathcal{I}$ denote the sets of users and items, respectively. For each user $u \in \mathcal{U}$, we have an interaction sequence $\mathcal{S}_u = [s_1, s_2, \ldots, s_m]$ where each $s_t$ represents an interaction with an item at time $t$. Our goal is to predict the next item that user $u$ will interact with based on their historical sequence.

Let $h_t^k \in \mathbb{R}^d$ denote the representation of interaction $s_t$ obtained after the k-th block of a transformer-based recommender model. We define $H_k(\mathcal{S}_u) \in \mathbb{R}^d$ as the aggregated embedding of the entire interaction sequence $\mathcal{S}_u$ produced by the $k$-th block, which serves as an internal user representation.

### LLM-Based User Profile Generation

We prompt a pre-trained LLM (Gemma-2-9b) using users' interaction history from the training data, providing available textual information about items such as product titles, descriptions, genres, and metadata. The LLM characterizes user preferences and extracts behavioral insights through structured prompting strategies including long-form descriptions and organized profiles with specific sections for preferences, behavioral patterns, and engagement trends.

We obtain vector representations of the generated textual profiles using the E5-large encoder: $E: \text{Profile} \rightarrow \mathbb{R}^e$. Since the embedding space typically differs from recommender models, we apply UMAP dimensionality reduction to preserve user relationships and a linear transformation $T: \mathbb{R}^e \to \mathbb{R}^d$ to align textual and recommender spaces.

### Knowledge Distillation to Sequential Recommender Models

We investigate multiple aggregation strategies to construct internal user representations $H_k(\mathcal{S}_u)$ from transformer hidden states:

**Average pooling**: 
```math
H_k(\mathcal{S}_u) = \frac{1}{m} \sum_{t=1}^{m} h_t^k
```

**Exponential weighting**: 
```math
H_k(\mathcal{S}_u) = \sum_{t=1}^{m} \left( \frac{\exp(\gamma \cdot t)}{\sum_{j=1}^{m} \exp(\gamma \cdot j)} \right) \cdot h_t^k
``` 
where $\gamma$ controls the decay rate.

**Learnable attention**: 
```math
H_k(\mathcal{S}_u) = \sum_{t=1}^{m} \left( \frac{\exp(w^\top h_t^k + b)}{\sum_{j=1}^{m} \exp(w^\top h_j^k + b)} \right) \cdot h_t^k
``` 
where $w$ and $b$ are learnable parameters.

### Two-Phase Training Process

**Distillation Stage**: We optimize both the main recommendation loss and auxiliary reconstruction loss:

```math
\text{Loss} = \alpha \cdot \text{Loss}_{distil} + (1 - \alpha) \cdot \text{Loss}_{model}
```

where $\text{Loss}_{distil} = \text{MSE}(H_k(\mathcal{S}_u), T(E(P(u))))$ aligns internal representations with LLM profiles. Since $\text{Loss}_{distil}$ is often much smaller than $\text{Loss}_{model}$, we introduce dynamic scaling: $\beta = \text{sg}\left(\frac{\text{Loss}_{model}}{\text{Loss}_{distil}}\right)$.

**Fine-tuning Stage**: We remove the auxiliary task and train exclusively on next-item prediction to refine recommendation quality while preserving distilled knowledge.

The method is model-agnostic, requires no architectural modifications, and eliminates inference-time LLM dependencies while achieving 60× faster inference compared to LLM-based approaches.

## Repository Structure

```
LLMProfileDistillation/
├── profile_generation/          # Phase 1: LLM-based user profile generation
│   ├── src/
│   │   ├── datasets/           # Dataset handlers
│   │   └── prompts/            # LLM prompts for profile generation
│   ├── encode.py               # Main script for profile generation
│   ├── encode_descriptions.py  # Script for embedding generation
│   └── README.md               # Detailed guide for profile generation
├── knowledge_transfer/          # Phase 2: Sequential recommendation with knowledge distillation
│   ├── src/
│   │   ├── models/             # Recommendation model implementations
│   │   ├── training.py         # Training script
│   │   ├── evaluation.py       # Evaluation utilities
│   │   └── data_processing.py  # Data preprocessing
│   ├── experiments/
│   │   └── configs/            # Configuration files for experiments
│   ├── data/                   # Dataset storage
│   └── requirements.txt        # Python dependencies
└── README.md                   # This file
```

## Datasets

We evaluate our method on four diverse datasets spanning different domains, scales, and textual richness to ensure comprehensive evaluation across various recommendation scenarios.

| Dataset | Domain | Textual Data | #Users | #Items | #Interactions | Avg.Interactions/User | Density |
|---------|--------|--------------|--------|--------|---------------|----------------------|---------|
| **Beauty** | Product Reviews | Rich | 70,996 | 39,116 | 436,309 | 6.145 | 0.00304 |
| **ML-20M** | Movies | Poor | 137,165 | 13,132 | 19,933,088 | 143.929 | 0.01096 |
| **Kion** | Movies | Rich | 16,797 | 5,626 | 287,698 | 17.128 | 0.00055 |
| **Amazon M2** | E-Commerce | Rich | 616,502 | 334,060 | 3,651,542 | 5.923 | 0.00002 |

Each dataset should follow this structure:

```
data/
├── <dataset_name>/
│   ├── raw/
│   │   ├── users.csv
│   │   ├── items.csv
│   │   └── interactions.csv
│   └── processed/
```

## Main Results

Our method consistently outperforms baseline approaches across all datasets:

### Performance Comparison: SASRec vs SASRec + LLM Distillation

| Dataset | Model | Recall@10 | NDCG@10 | Improvement |
|---------|--------|-----------|---------|-------------|
| **Beauty** | SASRec | 0.0217 ± 0.0013 | 0.0106 ± 0.0004 | - |
| | SASRec + LLM Distillation | **0.0228 ± 0.0004** | **0.0111 ± 0.0002** | +5.20% / +4.90% |
| **ML-20M** | SASRec | 0.0781 ± 0.0093 | 0.0453 ± 0.0062 | - |
| | SASRec + LLM Distillation | **0.0819 ± 0.0019** | **0.0479 ± 0.0007** | +4.74% / +5.62% |
| **Kion** | SASRec | 0.1135 ± 0.0009 | 0.0585 ± 0.0009 | - |
| | SASRec + LLM Distillation | **0.1145 ± 0.0007** | **0.0597 ± 0.0005** | +0.94% / +2.02% |
| **Amazon M2** | SASRec | 0.5373 ± 0.0123 | 0.3647 ± 0.0071 | - |
| | SASRec + LLM Distillation | **0.5414 ± 0.0051** | **0.3761 ± 0.0041** | +0.75% / +3.14% |

Our method consistently improves SASRec performance across all datasets, with particularly notable gains on Beauty and ML-20M datasets. The improvements demonstrate the effectiveness of LLM-derived user knowledge for enhancing sequential recommendation quality.

### Performance Comparison: BERT4Rec vs BERT4Rec + LLM Distillation

| Dataset | Model | Recall@10 | NDCG@10 | Improvement |
|---------|--------|-----------|---------|-------------|
| **Beauty** | BERT4Rec | 0.0102 ± 0.0004 | 0.0051 ± 0.0004 | - |
| | BERT4Rec + LLM Distillation | **0.0126 ± 0.0008** | **0.0061 ± 0.0005** | +23.53% / +19.61% |
| **ML-20M** | BERT4Rec | 0.1088 ± 0.0008 | 0.0623 ± 0.0001 | - |
| | BERT4Rec + LLM Distillation | **0.1099 ± 0.0029** | **0.0628 ± 0.0014** | +1.01% / +0.80% |
| **Kion** | BERT4Rec | 0.1101 ± 0.0018 | 0.0574 ± 0.0011 | - |
| | BERT4Rec + LLM Distillation | **0.1149 ± 0.0008** | **0.0596 ± 0.0009** | +4.36% / +3.83% |
| **Amazon M2** | BERT4Rec | 0.4230 ± 0.0013 | 0.2699 ± 0.0018 | - |
| | BERT4Rec + LLM Distillation | **0.4267 ± 0.0019** | **0.2727 ± 0.0020** | +0.87% / +1.04% |

BERT4Rec shows even more dramatic improvements, especially on the Beauty dataset with over 20% gains. This suggests our distillation approach is particularly effective for models that struggle with baseline performance, helping to bridge architectural limitations through LLM-derived semantic understanding.

### Comparison with State-of-the-Art LLM-based Method

| Method | Beauty | ML-20M | Kion | Amazon M2 |
|--------|--------|--------|------|-----------|
| **IDGenRec** | **0.0114** | 0.0313 | 0.0452 | 0.1001 |
| **SASRec** | 0.0106 | 0.0453 | 0.0585 | 0.3647 |
| **SASRec + LLM Distillation** | 0.0111 | **0.0479** | **0.0597** | **0.3761** |

*NDCG@10 scores. Bold indicates best performance per dataset.*

While IDGenRec achieves the highest performance on Beauty, our approach outperforms it on three out of four datasets. Crucially, our method maintains competitive quality while offering significant computational advantages, as shown in the efficiency comparison below.

### Training and Inference Efficiency Comparison

| Method | Beauty | ML-20M | Kion | Amazon M2 |
|--------|--------|--------|------|-----------|
| | Training / Inference | Training / Inference | Training / Inference | Training / Inference |
| **IDGenRec** | 37.11s / 120.38s | 132.12s / 840.01s | 80.04s / 342.03s | 1529.45s / 7492.11s |
| **SASRec** | 25.56s / 2.37s | 68.30s / 4.37s | 30.93s / 1.12s | 663.26s / 41.07s |
| **SASRec + LLM Distillation** | 26.92s / 2.37s | 71.34s / 4.37s | 31.81s / 1.12s | 833.36s / 41.07s |

*Training time per epoch and inference time for test set in seconds. Our method maintains identical inference speed to vanilla SASRec while being 50-180× faster than IDGenRec.*

The efficiency comparison reveals the key advantage of our approach: we achieve competitive or superior performance to state-of-the-art LLM-based methods while maintaining production-ready inference speeds. Our method adds minimal training overhead (5-25%) but eliminates the massive inference bottleneck that makes IDGenRec impractical for real-time applications.


## Configuration

Configuration files in `knowledge_transfer/experiments/configs/` control model architecture (SASRec, BERT4Rec, etc.), training hyperparameters, knowledge distillation settings, and evaluation metrics.

**Profile Generation Parameters**: Use `--dataset` for dataset name, `--llm` for LLM choice (openai, gemma2-9b), `--prompts-type` for profile format (long, short), and `--long-gen-strategy` for aggregation strategy (agg_after, agg_with).

### Hyperparameter Analysis

#### Loss Balancing Parameters (Beauty Dataset)

| Dynamic Scaling (β) | α | NDCG@10 |
|---------------------|---|---------|
| No | 0.4 | 0.0105 |
| No | 0.6 | 0.0108 |
| No | 0.8 | 0.0109 |
| **Yes** | **0.4** | **0.0111** |
| Yes | 0.6 | 0.0106 |
| Yes | 0.8 | 0.0106 |

*Dynamic scaling (β) with α=0.4 achieves optimal performance.*

#### Optimal User Representation Strategies

| Dataset | Best Aggregation Method |
|---------|------------------------|
| **Beauty** | Mean Pooling |
| **Kion** | Mean Pooling |
| **ML-20M** | Exponential Weighting |
| **Amazon M2** | Exponential Weighting |

*Choice depends on user behavior patterns: stable preferences favor mean pooling, while evolving preferences benefit from exponential weighting.*

## Detailed Usage

For detailed instructions on each component, see `profile_generation/README.md` for profile generation and `knowledge_transfer/README.md` for knowledge transfer training.

## Cold-Start Performance

Our method shows particularly strong performance for cold-start scenarios, achieving up to 72% improvement for items with limited training data. The approach demonstrates significant improvements when user history is sparse and enables better generalization across item categories through LLM-derived profiles.

### Cold-Start Experimental Results

| Dataset | Model | Recall@10 | NDCG@10 | Improvement |
|---------|--------|-----------|---------|-------------|
| **Beauty** | SASRec (Base) | 0.01477 ± 0.0012 | 0.0076 ± 0.0006 | - |
| | SASRec + LLM Distillation | **0.01614 ± 0.0006** | **0.0079 ± 0.0003** | +9.28% / +3.95% |
| **ML-20M** | SASRec (Base) | 0.001337 ± 0.0003 | 0.0007 ± 0.0001 | - |
| | SASRec + LLM Distillation | **0.0023 ± 0.0006** | **0.0012 ± 0.0003** | +72.03% / +71.43% |

*Results for cold-start scenarios where 40% of test items have limited training data (6-9 interactions) or are completely removed from training.*

### Cold-Start Dataset Statistics

| Dataset | #Users | #Items | #Interactions | #Items Removed | #Items Reduced | #Test Users with Cold Items |
|---------|--------|--------|---------------|----------------|----------------|-----------------------------|
| **Beauty** | 32,197 | 15,693 | 251,277 | 4,151 | 3,978 | 10,386 |
| **ML-20M** | 137,165 | 13,132 | 13,094,946 | 3,270 | 3,278 | 31,953 |


## Reproduction

To reproduce our results, follow these three main steps: install dependencies, generate user profiles with LLMs, and train models with knowledge distillation.

### Install Dependencies

```bash
pip install -r knowledge_transfer/requirements.txt
```

### Generate User Profiles

```bash
cd profile_generation

# Generate user profiles using LLM
python encode.py --dataset='beauty' \
                 --dataset-path='../data/amazon_beauty' \
                 --llm='gemma2-9b' \
                 --descriptions-path='./output' \
                 --long-gen-strategy="agg_after" \
                 --max_output_tokens=1000

# Convert profiles to embeddings
python encode_descriptions.py --embedder="e5" \
                              --descriptions-path="data/descriptions.json" \
                              --embeddings-path="data/embeddings.json"
```

### Train Models

```bash
cd knowledge_transfer

# Train enhanced model with LLM distillation
python -m src.training --config experiments/configs/sasrec_llm.yaml

# Train baseline model for comparison
python -m src.training --config experiments/configs/sasrec.yaml
```

All configuration files and detailed instructions are available in the respective README files within each directory.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{llm_profile_distillation2025,
  title={Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
