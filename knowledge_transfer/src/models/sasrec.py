import torch
import torch.nn as nn
import numpy as np

from src.models.utils import mean_weightening


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRec(nn.Module):
    def __init__(self, item_num, maxlen=50, hidden_units=64,
                 num_blocks=2, num_heads=2, dropout_rate=0.2,
                 initializer_range=0.02, add_head=True):
        super(SASRec, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate)
            )
            self.forward_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate)
            )

        # Инициализация параметров
        self.apply(self._init_weights)
        self.profile_transform = nn.Linear(128, self.hidden_units)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight.data[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, user_profile_emb=None):
        seqs = self.item_emb(input_ids)
        seqs *= self.hidden_units ** 0.5

        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, dtype=torch.long,
                                 device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        seqs += self.pos_emb(positions)

        if user_profile_emb is not None:
            seqs += user_profile_emb.unsqueeze(1)

        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=seqs.device)
        )

        for i in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_outputs
            seqs = seqs.transpose(0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs)
        reconstruction_input = mean_weightening(outputs)

        if self.add_head:
            outputs = torch.matmul(
                outputs, self.item_emb.weight.transpose(0, 1)
            )

        return outputs, reconstruction_input

    def aggregate_profile(self, user_profile_emb):
        """
        user_profile_emb: [batch_size, emb_dim]  или  [batch_size, K, emb_dim]
        Возвращает: [batch_size, hidden_units] (если use_down_scale=True) либо [batch_size, emb_dim].
        """
        if user_profile_emb is None:
            return None

        if user_profile_emb.dim() == 2:
            return self.profile_transform(user_profile_emb)
        raise Exception('aggregate_profile: Not Implemented error')