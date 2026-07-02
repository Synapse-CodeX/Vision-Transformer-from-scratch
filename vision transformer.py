import torch
import torch.nn as nn 

class PatchEmbedding(nn.Module):
    """
    Slices a 2D image into flattened 1D patches and projects them into a continuous embedding space.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # Input shape: [B, C, H, W] -> [B,3,224,224]
        x = self.proj(x)          # -> [B, embed_dim, H/patch_size, W/patch_size] -> [B,768,14,14]
        x = x.flatten(2)          # -> [B, embed_dim, Num_Patches] -> [B,768,196]
        x = x.transpose(1, 2)     # -> [B, Num_Patches, embed_dim] -> [B,196,768]
        return x


class MultiHeadAttention(nn.Module):
    """
    Computes Scaled Dot-Product Multi-Head Self-Attention.

    Input  : [B, N, C]
             B = Batch Size
             N = Number of Tokens (196 Patches + 1 CLS = 197)
             C = Embedding Dimension (768)

    Output : [B, N, C]
    """

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        # 768 / 12 = 64
        self.head_dim = embed_dim // num_heads
        # = 1 / sqrt(head_dim)
        self.scale = self.head_dim ** -0.5
        # Safety check
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be perfectly divisible by num_heads"

        # ---------------------------------------------------
        # One Linear layer simultaneously creates
        #
        # Query  : 768
        # Key    : 768
        # Value  : 768
        #
        # Total output = 2304
        #
        # Shape:
        # [B,N,768]
        #      ↓
        # [B,N,2304]
        # ---------------------------------------------------
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)

        # Dropout after Softmax Attention
        self.attn_drop = nn.Dropout(dropout)

        # Final projection after merging all heads
        # [B,N,768] → [B,N,768]
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Output Dropout
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # [2,197,768]
        B, N, C = x.shape

        # ==========================================================
        # STEP 1 : Generate Query, Key and Value
        #
        # [B,N,768]
        #       ↓
        # Linear
        #       ↓
        # [B,N,2304]
        #
        # 2304 = 3 × 12 × 64
        # ==========================================================
        qkv = self.qkv(x)

        # ==========================================================
        # STEP 2 : Split into
        #
        # 3   -> Q,K,V
        # 12  -> Attention Heads
        # 64  -> Features per Head
        #
        # [B,N,2304]
        #       ↓
        # reshape
        #       ↓
        # [B,N,3,12,64]
        # ==========================================================
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # ==========================================================
        # STEP 3 : Rearrange Dimensions
        #
        # Before
        # [B,N,3,12,64]
        #
        # After permute(2,0,3,1,4)
        #
        # [3,B,12,N,64]
        #
        # Dimension Meaning
        #
        # 0 → Q,K,V
        # 1 → Batch
        # 2 → Heads
        # 3 → Tokens
        # 4 → Head Dimension
        # ==========================================================
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # ==========================================================
        # STEP 4 : Separate Q, K and V
        #
        # Each has shape
        #
        # [B,12,N,64]
        #
        # Example
        #
        # [2,12,197,64]
        # ==========================================================
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ==========================================================
        # STEP 5 : Compute Similarity Matrix
        #
        # K is transposed
        #
        # K
        # [B,12,N,64]
        #
        # ↓ transpose(-2,-1)
        #
        # [B,12,64,N]
        #
        # Matrix Multiplication
        #
        # Q @ Kᵀ
        #
        # [B,12,N,64]
        #        @
        # [B,12,64,N]
        #
        # =
        #
        # [B,12,N,N]
        #
        # Every token now has similarity
        # with every other token.
        # ==========================================================
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ==========================================================
        # STEP 6 : Scale
        #
        # Multiply by
        #
        # 1 / sqrt(64)
        #
        # to stabilize Softmax.
        #
        # Shape remains
        #
        # [B,12,N,N]
        # ==========================================================

        # ==========================================================
        # STEP 7 : Softmax
        #
        # Converts similarity scores
        # into probabilities.
        #
        # Shape
        #
        # [B,12,N,N]
        # ==========================================================
        attn = attn.softmax(dim=-1)

        # ==========================================================
        # STEP 8 : Dropout
        #
        # Shape unchanged
        #
        # [B,12,N,N]
        # ==========================================================
        attn = self.attn_drop(attn)

        # ==========================================================
        # STEP 9 : Aggregate Context
        #
        # Attention @ Value
        #
        # [B,12,N,N]
        #        @
        # [B,12,N,64]
        #
        # =
        #
        # [B,12,N,64]
        #
        # Every token becomes a weighted
        # combination of all Value vectors.
        # ==========================================================
        x = attn @ v

        # ==========================================================
        # STEP 10 : Bring Tokens before Heads
        #
        # Before
        #
        # [B,12,N,64]
        #
        # After transpose(1,2)
        #
        # [B,N,12,64]
        # ==========================================================
        x = x.transpose(1, 2)

        # ==========================================================
        # STEP 11 : Merge Heads
        #
        # 12 × 64 = 768
        #
        # [B,N,12,64]
        #
        # ↓ reshape
        #
        # [B,N,768]
        # ==========================================================
        x = x.reshape(B, N, C)

        # ==========================================================
        # STEP 12 : Final Linear Projection
        #
        # [B,N,768]
        #
        # ↓
        #
        # [B,N,768]
        # ==========================================================
        x = self.proj(x)

        # ==========================================================
        # STEP 13 : Output Dropout
        #
        # Final Output
        #
        # [B,N,768]
        # ==========================================================
        x = self.proj_drop(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    A single Vision Transformer layer featuring Pre-Layer Normalization and residual paths.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention Block (with Pre-LN residual path)
        x = x + self.attn(self.norm1(x))
        # MLP Block (with Pre-LN residual path)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    The full Vision Transformer (ViT) architecture based on the original 2020 paper.
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        dropout=0.1
    ):
        super().__init__()
        
        # Initialize patch embedding pipeline
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Core learnable parameters: Classification token and 1D positional tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Sequence of structural transformer layers
        self.blocks = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        
        # Prediction head
        self.head = nn.Linear(embed_dim, num_classes)

        # Execute specialized initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using truncated normal and constant zeros to stabilize convergence.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Transform spatial image inputs to 1D patch sequences
        x = self.patch_embed(x) # -> [B, Num_Patches, embed_dim]

        # 2. Replicate CLS tokens for the batch and prepend to sequences
        cls_tokens = self.cls_token.expand(B, -1, -1) # -> [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)         # -> [B, Num_Patches + 1, embed_dim]

        # 3. Superimpose positional data
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Run through deep representation layers
        x = self.blocks(x)
        x = self.norm(x)

        # 5. Route the isolated classification token vector to the target linear head
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)


# --- Quick Functional Verification Unit ---
if __name__ == "__main__":
    # Generate mock evaluation batch representing 2 RGB 224x224 images
    mock_batch = torch.randn(2, 3, 224, 224)
    
    # Initialize complete ViT-Base architecture configuration
    vit_model = VisionTransformer(img_size=224, patch_size=16, num_classes=1000)
    
    # Forward evaluation pass
    logits = vit_model(mock_batch)
    print(f"Target verification successful!")
    print(f"Input Image Tensor Batch Shape:  {mock_batch.shape}")
    print(f"Output Class Prediction Shape:   {logits.shape}")
