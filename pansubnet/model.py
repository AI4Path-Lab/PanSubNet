"""
PanSubNet model.

Hierarchy
---------
1. ``SpatialAwareAttention`` - self-attention over the nuclei inside a single
   patch, with an optional spatial-distance bias between cell centroids. Emits a
   CLS token summarising the patch's cellular composition.
2. ``TensorFusion``          - bilinear (outer-product) fusion of the cell CLS
   token with the projected patch (foundation-model) embedding.
3. ``PatchProcessor``        - runs 1 + 2 for one patch and returns its
   representation.
4. ``AttentionMIL`` / ``CASii_MB`` - aggregates the per-patch representations of
   a whole slide into a slide-level logit.
5. ``WSIClassifier``         - ties everything together and streams patches to
   the GPU one at a time to keep memory bounded.
"""

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAwareAttention(nn.Module):
    """Single-head self-attention over the cells of one patch."""

    def __init__(self, cell_dim=1280, hidden_dim=768, distance_metric="euclidean",
                 use_spatial_bias=True, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.distance_metric = distance_metric
        self.use_spatial_bias = use_spatial_bias

        self.query_proj = nn.Linear(cell_dim, hidden_dim)
        self.key_proj = nn.Linear(cell_dim, hidden_dim)
        self.value_proj = nn.Linear(cell_dim, hidden_dim)

        self.pre_norm = nn.LayerNorm(cell_dim)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # Learnable CLS token that aggregates the patch's cells.
        self.cls_token = nn.Parameter(torch.randn(1, 1, cell_dim))
        self.dropout = nn.Dropout(dropout)

    def compute_distance_matrix(self, centroids):
        """Pairwise distances between cell centroids.

        Args:
            centroids: ``[B, N, 2]``
        Returns:
            ``[B, N, N]`` distance matrix.
        """
        x = centroids.unsqueeze(2)  # [B, N, 1, 2]
        y = centroids.unsqueeze(1)  # [B, 1, N, 2]
        if self.distance_metric == "euclidean":
            return torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
        if self.distance_metric == "manhattan":
            return torch.sum(torch.abs(x - y), dim=-1)
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def forward(self, patch_coords, cell_embeddings, centroids):
        """
        Args:
            patch_coords: ``[2]`` top-left coordinate of the patch (CLS "location").
            cell_embeddings: ``[B, N, cell_dim]``
            centroids: ``[B, N, 2]``
        Returns:
            (cls_embedding ``[B, hidden_dim]``, attention_probs ``[B, N+1, N+1]``)
        """
        batch_size = cell_embeddings.shape[0]

        cell_embeddings = self.pre_norm(cell_embeddings)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, cell_embeddings], dim=1)  # [B, N+1, cell_dim]

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float32, device=x.device)
        )

        if self.use_spatial_bias:
            cls_centroid = patch_coords.unsqueeze(0)
            all_centroids = torch.cat([cls_centroid, centroids], dim=1)  # [B, N+1, 2]
            distance_matrix = self.compute_distance_matrix(all_centroids)
            # Nearer cells attend more strongly.
            attn_scores = attn_scores - distance_matrix

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = self.post_norm(context)

        cls_token_embedding = context[:, 0, :]
        return cls_token_embedding, attn_probs


class TensorFusion(nn.Module):
    """Outer-product fusion of two vectors, projected back to ``output_dim``."""

    def __init__(self, dim1=768, dim2=768, output_dim=768):
        super().__init__()
        self.projection = nn.Linear((dim1 + 1) * (dim2 + 1), output_dim)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        ones = torch.ones(batch_size, 1, device=x1.device)
        x1_aug = torch.cat([x1, ones], dim=1)
        x2_aug = torch.cat([x2, ones], dim=1)
        fusion = torch.bmm(x1_aug.unsqueeze(2), x2_aug.unsqueeze(1))
        fusion = fusion.view(batch_size, -1)
        return self.projection(fusion)


class PatchProcessor(nn.Module):
    """Produce a single representation vector for one patch."""

    def __init__(self, patch_dim=1536, cell_dim=1280, hidden_dim=768,
                 distance_metric="euclidean", use_spatial_bias=True,
                 use_cell_ratios=True, use_patch_embeddings=True,
                 patch_embeddings_only=False):
        super().__init__()
        self.use_cell_ratios = use_cell_ratios
        self.use_patch_embeddings = use_patch_embeddings
        self.patch_embeddings_only = patch_embeddings_only

        if not self.patch_embeddings_only:
            self.cell_attention = SpatialAwareAttention(
                cell_dim=cell_dim, hidden_dim=hidden_dim,
                distance_metric=distance_metric, use_spatial_bias=use_spatial_bias,
            )

        if self.use_patch_embeddings:
            self.patch_projection = nn.Linear(patch_dim, hidden_dim)

        if not self.patch_embeddings_only:
            if self.use_cell_ratios:
                # + 5 PanNuke cell-type ratios.
                self.cls_projection = nn.Linear(hidden_dim + 5, hidden_dim)
            if self.use_patch_embeddings:
                self.fusion = TensorFusion(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, patch_coords, patch_embedding, cell_embeddings,
                cell_centroids, cell_ratios=None):
        if self.patch_embeddings_only:
            return self.patch_projection(patch_embedding), None

        cls_token, cell_probs = self.cell_attention(
            patch_coords, cell_embeddings, cell_centroids
        )

        if self.use_cell_ratios:
            cls_token = torch.cat((cls_token, cell_ratios), dim=1)
            cls_token = self.cls_projection(cls_token)

        if not self.use_patch_embeddings:
            return cls_token, cell_probs

        patch_proj = self.patch_projection(patch_embedding)
        return self.fusion(cls_token, patch_proj), cell_probs


class AttentionMIL(nn.Module):
    """Gated attention MIL (Ilse et al., 2018) with a multi-dim attention head."""

    def __init__(self, input_dim=768, hidden_dim1=512, hidden_dim2=256,
                 projection_dim=1, num_classes=1, dropout=0.1,
                 aggregation_method="norm"):
        super().__init__()
        self.hidden_dim1 = hidden_dim1
        self.projection_dim = projection_dim
        self.aggregation_method = aggregation_method

        self.feature_extractor = nn.Sequential(nn.Linear(input_dim, hidden_dim1), nn.ReLU())
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim1, hidden_dim2), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim1, hidden_dim2), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim2, projection_dim, bias=False)

        if self.aggregation_method == "learnable":
            self.projection_aggregator = nn.Linear(projection_dim, 1, bias=False)

        self.classifier = nn.Sequential(nn.Linear(hidden_dim1, num_classes))

    def forward(self, x):
        """
        Args:
            x: ``[B, num_patches, input_dim]``
        Returns:
            logits ``[B, num_classes]``, attention ``[B, 1, num_patches]``,
            slide embedding ``[B, hidden_dim1]``, projection matrix, and the
            per-patch gated features.
        """
        x = self.feature_extractor(x)

        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a_proj = self.attention_weights(a_v * a_u)  # [B, num_patches, projection_dim]

        if self.projection_dim == 1:
            a = a_proj
        elif self.aggregation_method == "norm":
            a = torch.norm(a_proj, dim=2, keepdim=True)
        elif self.aggregation_method == "learnable":
            a = self.projection_aggregator(a_proj)
        else:
            raise ValueError(f"Unknown aggregation_method: {self.aggregation_method}")

        a = a.permute(0, 2, 1)          # [B, 1, num_patches]
        a = F.softmax(a, dim=2)

        m = torch.matmul(a, x).view(-1, self.hidden_dim1)  # [B, hidden_dim1]
        y_prob = self.classifier(m)
        w = self.attention_weights.weight.transpose(0, 1)  # [hidden_dim2, projection_dim]

        return y_prob, a, m, w, a_v * a_u


class CASiiHead(nn.Module):
    """One cross-attention head between slide patches and a learned key set."""

    def __init__(self, inputd=1024, hd=512, k=10, A_dim=3200, tau=1):
        super().__init__()
        self.hd = hd
        self.k = k
        self.tau = tau
        self.WQ = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WK = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WV = nn.Sequential(nn.Linear(inputd, hd), nn.ReLU())
        self.WA = nn.Linear(A_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(hd, 1))

    def metafusion(self, A):
        A = self.WA(A).squeeze(-1)
        topq, _ = torch.topk(A, int(self.k), dim=-1)
        botq, _ = torch.topk(A, int(self.k), dim=-1, largest=False)
        A = F.softmax(A / self.tau, dim=1).unsqueeze(2)
        return A, topq, botq

    def forward(self, x):
        query, key = x
        value = self.WV(query)
        query = self.WQ(query)
        key = self.WK(key)

        q_norm = F.normalize(query, p=2, dim=2)
        k_norm = F.normalize(key, p=2, dim=2).transpose(2, 1)

        A = torch.matmul(q_norm, k_norm)
        A, topq, botq = self.metafusion(A)

        A = A.permute(0, 2, 1)
        z = torch.matmul(A, value).view(-1, self.hd)
        return self.classifier(z), A


class CASii_MB(nn.Module):
    """Two-branch CASii MIL (low / high key sets)."""

    def __init__(self, inputd=768, hd=512, n_head=2, A_dims=(4000, 4000), k=1, tau=1):
        super().__init__()
        self.hd = hd
        self.k = k
        self.tau = tau
        self.n_head = n_head
        self.heads = nn.ModuleList(
            [CASiiHead(inputd=inputd, hd=hd, k=k, A_dim=d) for d in A_dims]
        )

    def forward(self, x):
        logits = torch.empty(1, self.n_head, device=x[0].device).float()
        atts = [None] * self.n_head
        for c in range(self.n_head):
            logits[0, c], atts[c] = self.heads[c]([x[0], x[c + 1]])
        y_hat = torch.topk(logits, 1, dim=1)[1]
        return y_hat, logits, atts


class WSIClassifier(nn.Module):
    """End-to-end slide classifier."""

    def __init__(self, patch_dim=1536, cell_dim=1280, hidden_dim1=768,
                 hidden_dim2=512, hidden_dim3=256, num_classes=2,
                 distance_metric="euclidean", use_spatial_bias=True, dropout=0.1,
                 use_cell_ratios=True, projection_dim=1, aggregation_method="norm",
                 use_patch_embeddings=True, patch_embeddings_only=False, mil="att",
                 device=None, lowkeysetlength=0, highkeysetlength=0):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cell_ratios = use_cell_ratios
        self.use_patch_embeddings = use_patch_embeddings
        self.patch_embeddings_only = patch_embeddings_only
        self.mil = mil

        self.patch_processor = PatchProcessor(
            patch_dim=patch_dim, cell_dim=cell_dim, hidden_dim=hidden_dim1,
            distance_metric=distance_metric, use_spatial_bias=use_spatial_bias,
            use_cell_ratios=use_cell_ratios, use_patch_embeddings=use_patch_embeddings,
            patch_embeddings_only=patch_embeddings_only,
        )

        if self.mil == "att":
            self.attention_mil = AttentionMIL(
                input_dim=hidden_dim1, hidden_dim1=hidden_dim2, hidden_dim2=hidden_dim3,
                num_classes=num_classes, dropout=dropout,
                aggregation_method=aggregation_method, projection_dim=projection_dim,
            )
        elif self.mil == "casii":
            self.attention_mil = CASii_MB(
                inputd=hidden_dim1, hd=hidden_dim2, n_head=2,
                A_dims=[lowkeysetlength, highkeysetlength], k=1, tau=1,
            )
        else:
            raise ValueError(f"Unknown mil: {mil}")

    def forward(self, x):
        """
        Args:
            x: for ``mil='att'`` a ``wsi_dict`` mapping ``(x, y) -> patch_data``;
               for ``mil='casii'`` a tuple ``(wsi_dict, lowkeyset, highkeyset)``.
        Returns:
            ``mil='att'``  -> (logits, patch_attention, slide_embedding, W,
                               gated_features, patch_embeddings, cellpatchattn)
            ``mil='casii'`` -> (y_hat, logits, head_attentions, patch_embeddings)
        """
        if self.mil == "casii":
            wsi_dict, lowkeyset, highkeyset = x
        else:
            wsi_dict = x

        patch_embeddings = []
        coords_list = []
        cellpatchattn = {}

        for j, (coords, patch_data) in enumerate(wsi_dict.items()):
            patch_coords = patch_emb = cell_embs = cell_cents = cell_ratios = None
            try:
                patch_coords = patch_data["patch_coords"].to(self.device, non_blocking=True)
                patch_emb = patch_data["patch_embeddings"].to(self.device, non_blocking=True)
                cell_embs = patch_data["cell_embeddings"].to(self.device, non_blocking=True)
                cell_cents = patch_data["cell_centroids"].to(self.device, non_blocking=True)
                cell_ratios = patch_data.get("cell_types", None)
                if cell_ratios is not None:
                    cell_ratios = cell_ratios.to(self.device, non_blocking=True)

                if self.use_cell_ratios and cell_ratios is not None:
                    patch_repr, cell_probs = self.patch_processor(
                        patch_coords, patch_emb, cell_embs, cell_cents, cell_ratios
                    )
                else:
                    patch_repr, cell_probs = self.patch_processor(
                        patch_coords, patch_emb, cell_embs, cell_cents
                    )

                patch_embeddings.append(patch_repr)
                coords_list.append(coords)
                cellpatchattn[coords] = {
                    "cellattention": cell_probs,
                    "cellcentroids": cell_cents.cpu(),
                }
            except RuntimeError as err:
                if "out of memory" in str(err):
                    print(f"OOM while processing patch {j} ({coords}) - skipping.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            finally:
                del patch_coords, patch_emb, cell_embs, cell_cents, cell_ratios

        if not patch_embeddings:
            raise ValueError("No valid patches processed for this WSI.")

        patch_embeddings = torch.cat(patch_embeddings, dim=0).unsqueeze(0)

        if self.mil == "casii":
            y_hat, logits, atts = self.attention_mil((patch_embeddings, lowkeyset, highkeyset))
            return y_hat, logits, atts, patch_embeddings

        y_prob, attn_weights, slide_embeddings, w_matrix, samples = self.attention_mil(patch_embeddings)
        attn_flat = attn_weights.squeeze(0).squeeze(0)
        for i, coords in enumerate(coords_list):
            cellpatchattn[coords]["patch_attention"] = attn_flat[i].item()
        return y_prob, attn_weights, slide_embeddings, w_matrix, samples, patch_embeddings, cellpatchattn
