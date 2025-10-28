import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size=4, emb_dim=16, k=5, num_pos=3, out_dim=32, n_heads=2, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_pos, k, emb_dim))
        flat = k * emb_dim
        layer = nn.TransformerEncoderLayer(d_model=flat, nhead=n_heads, dim_feedforward=128,
                                           batch_first=True, dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())
    def forward(self, seq):
        e = self.emb(seq) + self.pos_emb
        e = e.view(e.size(0), e.size(1), -1)
        e = self.encoder(e)
        return self.fc(e.mean(dim=1))

class ReadFeatureEncoder(nn.Module):
    def __init__(self, input_dim=9, seq_dim=32, hidden_dim=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.fc_seq = nn.Linear(seq_dim, hidden_dim)
        self.fc_reads = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Sigmoid()
        self.mlp = nn.Sequential(nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim), nn.ReLU())
    def forward(self, reads, seq_vec):
        seq_h = self.fc_seq(seq_vec).unsqueeze(1)
        read_h = self.fc_reads(reads)
        return self.mlp(read_h * self.gate(seq_h))

class AttentionPooling(nn.Module):
    def __init__(self, in_dim, att_dim=64):
        super().__init__()
        self.V = nn.Linear(in_dim, att_dim)
        self.U = nn.Linear(in_dim, att_dim)
        self.w = nn.Linear(att_dim, 1)
    def forward(self, H, mask=None):
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
        a = self.w(A).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        att = torch.softmax(a, dim=1)
        M = torch.sum(att.unsqueeze(-1) * H, dim=1)
        return M, att

class SiteClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_enc = SequenceEncoder()
        self.read_enc = ReadFeatureEncoder()
        self.ln1 = nn.LayerNorm(64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           batch_first=True, dropout=0.1, activation="gelu")
        self.read_transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.ln2 = nn.LayerNorm(64)
        self.pool = AttentionPooling(64, 64)
        self.head = nn.Linear(64, 1)
    def forward(self, reads, seq, mask=None):
        seq_vec = self.seq_enc(seq)
        x = self.read_enc(reads, seq_vec)
        x = self.ln1(x)
        x = self.read_transformer(x)
        x = self.ln2(x)
        pooled, att = self.pool(x, mask)
        return self.head(pooled), att

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=1.25, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()
