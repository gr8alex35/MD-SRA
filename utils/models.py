import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Text + side feature fusion
# -----------------------------
class TextFeatureFusion(nn.Module):
    def __init__(self, feat_hidden=64, out_dim=512):
        super().__init__()
        self._feature_ln = None
        self.feature_fc = nn.LazyLinear(feat_hidden)
        self.fusion_fc1 = nn.LazyLinear(256)
        self.fusion_fc2 = nn.Linear(256, out_dim)

    def forward(self, text_emb, feature):
        if self._feature_ln is None:
            self._feature_ln = nn.LayerNorm(feature.size(1)).to(feature.device)
        f = self._feature_ln(feature)
        f = F.relu(self.feature_fc(f))
        x = torch.cat([text_emb, f], dim=1)
        x = F.relu(self.fusion_fc1(x))
        x = self.fusion_fc2(x)             # [B,512]
        return x


# -----------------------------
# Cross-Attention Block
# -----------------------------
class CrossAttnBlock(nn.Module):
    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1  = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, q_vec, kv_vec):
        q = q_vec.unsqueeze(1)   # [B,1,512]
        kv = kv_vec.unsqueeze(1) # [B,1,512]
        z, _ = self.attn(q, kv, kv)
        z = self.ln1(z + q)
        z = z + self.ffn(z)
        return z.squeeze(1)      # [B,512]


# -----------------------------
# MDSRA
# -----------------------------
class MDSRA(nn.Module):
    def __init__(self, use_start=True, use_end=True, use_image=True,
                 n_class=3, cross_heads=8):
        super().__init__()
        self.use_start = use_start
        self.use_end   = use_end
        self.use_image = use_image

        if use_start:
            self.text_start_fusion = TextFeatureFusion(out_dim=512)
        if use_end:
            self.end_proj   = nn.LazyLinear(512)
        if use_image:
            self.image_proj = nn.LazyLinear(512)

        # Cross-Attn blocks
        self.cross_e2s = CrossAttnBlock(512, n_heads=cross_heads)  # END→START
        self.cross_e2i = CrossAttnBlock(512, n_heads=cross_heads)  # END→IMG

        # Classifier
        self.classifier = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, n_class)
        )

    def forward(self, start_text=None, start_feat=None,
                end_text=None, image_emb=None, return_intermediates=False):
        start_512 = end_512 = image_512 = None
        feats = []

        # End
        if self.use_end:
            end_512 = self.end_proj(end_text)
            feats.append(end_512)

        # Start
        if self.use_start:
            start_512 = self.text_start_fusion(start_text, start_feat)
            feats.append(start_512)

        # Image
        if self.use_image:
            image_512 = self.image_proj(image_emb)
            feats.append(image_512)

        # --- Cross-Attention ---
        if self.use_start and self.use_end and self.use_image:
            s_ctx = self.cross_e2s(end_512, start_512)  # START’ (E→S)
            i_ctx = self.cross_e2i(end_512, image_512)  # IMG’   (E→I)
            # Residual concat
            x = torch.cat([s_ctx, start_512, end_512, i_ctx, image_512], dim=1)
        else:
            x = torch.cat(feats, dim=1)

        return self.classifier(x)