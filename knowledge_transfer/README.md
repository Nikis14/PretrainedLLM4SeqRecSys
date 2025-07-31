# Knowledge Transfer for Sequential Recommendation

This module implements the knowledge distillation component of our LLM-driven user profile distillation method. It trains sequential recommender models (SASRec, BERT4Rec) to reconstruct LLM-generated user profiles through an auxiliary loss, effectively embedding semantic understanding into model parameters.

## Overview

The knowledge transfer process consists of two main phases:
1. **Distillation Stage**: Train with both recommendation loss and profile reconstruction loss
2. **Fine-tuning Stage**: Refine with recommendation loss only to optimize final performance

## Quick Start

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Training

Train a model with LLM profile distillation:
```bash
python -m src.training --config experiments/configs/sasrec_llm.yaml
```

Train baseline model for comparison:
```bash
python -m src.training --config experiments/configs/sasrec.yaml
```

### Evaluation

Evaluate trained models:
```bash
python -m src.evaluation --model-path outputs/sasrec_llm/model.pt --config experiments/configs/sasrec_llm.yaml
```

## Configuration

Configuration files are located in `experiments/configs/`. Key parameters:

### Model Configuration
- `model_type`: Model architecture (sasrec, bert4rec)
- `embedding_dim`: Model embedding dimension
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads

### Training Configuration
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate
- `weight_decay`: L2 regularization

### Distillation Configuration
- `use_distillation`: Enable/disable LLM profile distillation
- `distillation_alpha`: Weight for distillation loss (α)
- `use_dynamic_scaling`: Enable dynamic loss scaling (β)
- `aggregation_method`: User representation strategy (mean, exp_weight, attention)
- `distillation_layer`: Which transformer layer to use for user representation
- `profile_embeddings_path`: Path to LLM-generated profile embeddings

## Data Format

### Dataset Structure
```
data/
├── <dataset_name>/
│   ├── raw/
│   │   ├── users.csv          # User metadata
│   │   ├── items.csv          # Item metadata  
│   │   └── interactions.csv   # User-item interactions
│   └── processed/
│       ├── train_sequences.pkl    # Training sequences
│       ├── valid_sequences.pkl    # Validation sequences
│       ├── test_sequences.pkl     # Test sequences
│       ├── counts.pkl             # Item/user counts
│       └── mappings.pkl           # ID mappings
```

### Profile Embeddings Format
Profile embeddings should be stored as JSON:
```json
{
  "user_id_1": [0.1, 0.2, ..., 0.8],
  "user_id_2": [0.3, 0.1, ..., 0.9],
  ...
}
```

## Supported Models

### SASRec
Self-Attentive Sequential Recommendation using transformer encoder.

### BERT4Rec  
Bidirectional transformer for sequential recommendation with masked language modeling.

### Custom Models
Extend `src/models/base_model.py` to implement custom architectures with knowledge distillation support.

## Architecture Components

### User Representation Aggregation

**Mean Pooling**: Average all sequence representations
```math
H_k(S_u) = \frac{1}{m} \sum_{t=1}^{m} h_t^k
```

**Exponential Weighting**: Weight recent interactions more heavily
```math
H_k(S_u) = \sum_{t=1}^{m} \frac{\exp(\gamma \cdot t)}{\sum_{j=1}^{m} \exp(\gamma \cdot j)} \cdot h_t^k
```

**Learnable Attention**: Learn optimal weighting strategy
```math
H_k(S_u) = \sum_{t=1}^{m} \text{softmax}(w^T h_t^k + b) \cdot h_t^k
```

### Loss Functions

**Recommendation Loss**: Standard next-item prediction
```math
\mathcal{L}_{model} = \text{CrossEntropy}(\text{predictions}, \text{targets})
```

**Distillation Loss**: Profile reconstruction
```math
\mathcal{L}_{distil} = \text{MSE}(H_k(S_u), T(E(P(u))))
```

**Combined Loss**: Weighted combination with dynamic scaling
```math
\mathcal{L}_{total} = \alpha \cdot \beta \cdot \mathcal{L}_{distil} + (1-\alpha) \cdot \mathcal{L}_{model}
```

## Experimental Configuration

### Example Configuration Files

#### SASRec with LLM Distillation
```yaml
# experiments/configs/datasets/sasrec/beauty.yaml
seed: 42

model:
  model_name: SASRecLLM
  item_num: 0
  user_num: 0
  maxlen: 50
  hidden_units: 128
  num_blocks: 4
  num_heads: 2
  dropout_rate: 0.3
  initializer_range: 0.02
  add_head: true
  reconstruction_layer: -1
  weighting_scheme: mean
  weight_scale: 0.1
  use_down_scale: True
  use_upscale: False
  multi_profile: False
  multi_profile_aggr_scheme: mean

training:
  batch_size: 256
  epochs: 25
  learning_rate: 0.0005
  reconstruct_loss: MSE
  eval_every: 1
  model_dir: models/
  alpha: 0.4
  fine_tune_epoch: 12
  scale_guide_loss: true
  save_checkpoints: false

data:
  profile_train_sequences: /path/to/data/amazon_beauty/processed/train_sequences.pkl
  finetune_train_sequences: /path/to/data/amazon_beauty/processed/train_sequences.pkl
  valid_sequences: /path/to/data/amazon_beauty/processed/valid_sequences.pkl
  test_sequences: /path/to/data/amazon_beauty/processed/test_sequences.pkl
  mappings: /path/to/data/amazon_beauty/processed/mappings.pkl
  counts: /path/to/data/amazon_beauty/processed/counts.pkl
  user_profile_embeddings_files: /path/to/data/amazon_beauty/short_embedding_amazon_beauty_UMAP-128.json

experiment_name: "SASRec-LLM_Beauty"
```

## Evaluation Metrics

Standard recommendation metrics:
- **Recall@K**: Fraction of relevant items in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain considering ranking quality
- **Hit Rate@K**: Binary measure of successful recommendation

## Troubleshooting

### Common Issues

**CUDA Memory Errors**: Reduce batch size or model dimensions
**Slow Convergence**: Adjust learning rate or use learning rate scheduling  
**Poor Distillation**: Check profile embeddings alignment and loss balancing
**Overfitting**: Increase dropout or weight decay

### Debug Mode

Enable detailed logging:
```bash
python -m src.training --config config.yaml --debug --verbose
```

## Contributing

When adding new models or features:
1. Extend base classes in `src/models/`
2. Add configuration templates in `experiments/configs/`
3. Update evaluation metrics if needed
4. Add comprehensive tests

## References

For implementation details, see the main paper and `src/` source code documentation.