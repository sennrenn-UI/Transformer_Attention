# Transformer_Attention
Transfomer内のAttenrionについてのコード
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


tokens = tokens = ["彼", "は", "それ", "を", "見て", "笑った", "けど", "私", "は", "少し", "怖かった", "。"]

x = torch.randn(12, 12)

# ===== Step 2: Q, K, Vの作成（ =====
Q = x
K = x
V = x

# ===== Step 3: Attentionスコア計算 =====
d_k = Q.size(-1)  # 埋め込み次元の平方根で割る
scores = torch.matmul(Q, K.T) / (d_k ** 0.5)

# ===== Step 4: SoftmaxでAttention重みに変換 =====
attn_weights = F.softmax(scores, dim=-1)

# ===== Step 5: 出力ベクトルを計算 =====
output = torch.matmul(attn_weights, V)

# ===== Step 6: Attentionの可視化 =====
plt.figure(figsize=(6, 5))
sns.heatmap(attn_weights.detach().numpy(), annot=True, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu")
plt.xlabel("Key（見てる単語）")
plt.ylabel("Query（注目してる単語）")
plt.title("Self-Attention Weights")
plt.tight_layout()
plt.show()
