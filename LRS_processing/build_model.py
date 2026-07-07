import torch
import torch.nn as nn
import torch.nn.functional as F



class AuditoryEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, None))  # squash freq axis only
        )

    def forward(self, x):
        # x: (B, 1, n_mfcc, T)
        x = self.encoder(x)         # (B, 128, 1, T)
        x = x.squeeze(2)            # (B, 128, T)
        x = x.permute(0, 2, 1)      # (B, T, 128)
        return x



class VisualEncoder(nn.Module):
    """2D conv encoder for video frames: keeps time dimension T, outputs (B, T, 128)"""
    def __init__(self, in_channels=1, use_dropout=True, p=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input will be reshaped to (B*T, 1, H, W)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 96x96 -> 48x48

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x48 -> 24x24

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # -> (B*T, 128, 1, 1)
        )
        self.dropout = nn.Dropout(p) if use_dropout else nn.Identity()

    def forward(self, x):
        """
        x: (B, T, H, W) grayscale frames, float32, ideally normalized to [0,1]
        """
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)     # (B*T, 1, H, W)
        x = self.encoder(x)            # (B*T, 128, 1, 1)
        x = x.view(B, T, 128)          # (B, T, 128)
        x = self.dropout(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int, # default 128
        num_heads: int, # default 8
        use_q_proj: bool = True, # arg reserved for direct comparison of the features in cross-attn without applying weights: e.g. Q_out*W_Q -> Q_out if False
        use_k_proj: bool = True,
        use_v_proj: bool = True,
        return_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.return_attn = return_attn

        # Projections are optional. If disabled, we treat inputs as already in the right feature space.
        self.q_proj = nn.Linear(dim, dim) if use_q_proj else None
        self.k_proj = nn.Linear(dim, dim) if use_k_proj else None
        self.v_proj = nn.Linear(dim, dim) if use_v_proj else None

        self.out_proj = nn.Linear(dim, dim)

    def _shape_heads(self, x):  # (B,T,dim) -> (B,H,T,D)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def _build_temporal_keep_mask(
        Tq: int,
        Tk: int,
        device,
        mask_type: str = "none",  # "none" | "causal" | "causal_band"
        band: int = 0,            # radius; band=0 -> diagonal-only behavior 
    ):
        """
        Returns a bool mask of shape (1,1,Tq,Tk), True=keep.
        - "none": returns None
        - "causal": j <= i (no look ahead temporally)
        - "causal_band": j <= i AND |i - j| <= band (no look ahead temporally, and only attend to local frames)
        """
        if mask_type == "none":
            return None

        i = torch.arange(Tq, device=device).unsqueeze(1)  # (Tq,1)
        j = torch.arange(Tk, device=device).unsqueeze(0)  # (1,Tk)
        if mask_type == "causal":
            keep = (j <= i)
        elif mask_type == "causal_band":
            keep = (j <= i) & ((i - j).abs() <= band)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        # (1,1,Tq,Tk) broadcastable to (B,H,Tq,Tk)
        return keep.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        query,                     # (B,Tq,dim)  e.g., audio
        key,                       # (B,Tk,dim)  e.g., video
        value,                     # (B,Tk,dim)  typically same as key's source
        *,
        # External masks (orthogonal to temporal structure)
        attn_mask=None,            # broadcastable to (B,H,Tq,Tk); 1=keep, 0=mask
        key_padding_mask=None,     # (B,Tk) bool: True means PAD (mask it): to mask these q-k pairs that are actually padding positions
        # Temporal mask controls
        mask_type: str = "none",   # "none" | "causal" | "causal_band"
        band: int = 0,             # local window radius for banded masks
    ):
        B, Tq, _ = query.shape
        Bk, Tk, _ = key.shape
        assert B == Bk, "Batch size mismatch between Q and K/V"

        # Optional learnable projections
        Q_in = self.q_proj(query) if self.q_proj is not None else query
        K_in = self.k_proj(key)   if self.k_proj is not None else key
        V_in = self.v_proj(value) if self.v_proj is not None else value

        # shape to (B,H,T,head_dim)
        Q = self._shape_heads(Q_in)
        K = self._shape_heads(K_in)
        V = self._shape_heads(V_in)

        # scaled dot-product
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,H,Tq,Tk)

        # ---- Build & compose masks (assign mask type to use temporal masking, keep: a bool mask to preserve attn scores) ----
        keep = self._build_temporal_keep_mask(Tq, Tk, device=query.device,
                                              mask_type=mask_type, band=band)

        # key padding mask (B,Tk) -> (B,1,1,Tk), keep the real features (such as the padded frames in the aud/vis tensor due to length differences)
        if key_padding_mask is not None:
            k_keep = (~key_padding_mask).view(B, 1, 1, Tk)  # bool
            keep = k_keep if keep is None else (keep & k_keep)

        # external attn_mask expected as 1=keep, 0=mask
        if attn_mask is not None:
            a_keep = (attn_mask != 0)
            keep = a_keep if keep is None else (keep & a_keep)

        # if keep is not None:
        #     attn_scores = attn_scores.masked_fill_(~keep, float('-inf'))

        # keep: True=allowed after composing temporal (causal/causal_band) & key_padding & (optional) external
        if keep is None:
            keep = torch.ones_like(attn_scores, dtype=torch.bool)

        # Build query-valid mask (True where the query timestep is real): let the last padded frames of 
        # queries have value (deselect padding) so the (QK_t)V in this area is not NaN, we then turn these padded 
        # frames to be zero so they don't affect the results (or have small impact, since removing is the best way, 
        # but different lengths in one batch need extra computational power)

        # --- Query-valid mask (True where the query timestep is real) ---
        if key_padding_mask is not None and key_padding_mask.shape[-1] == Tq:
            # key_padding_mask: True=PAD, so invert -> True=valid
            q_valid = (~key_padding_mask).to(dtype=torch.bool, device=attn_scores.device)   # (B,Tq)
        else:
            # Fallback: treat all queries as valid (or build from an explicit lengths_q if you pass it)
            q_valid = torch.ones(B, Tq, dtype=torch.bool, device=attn_scores.device)        # (B,Tq)

        q_keep = q_valid.view(B, 1, Tq, 1)  # (B,1,Tq,1)

        # --- Compose final allowed mask ---
        if keep is None:
            keep = torch.ones_like(attn_scores, dtype=torch.bool)

        # --- Apply mask to scores ---
        allowed = keep & q_keep  # True = allowed Q-K pairs
        scores = attn_scores.masked_fill(~allowed, float('-inf'))

        # --- Row-safety guard (avoid softmax over all -inf) ---
        row_has_finite = torch.isfinite(scores).any(dim=-1, keepdim=True)  # (B,H,Tq,1)
        bad_rows = (~row_has_finite.squeeze(-1))                           # (B,H,Tq)
        bad_real = bad_rows & q_valid[:, None, :]                          # align q_valid
        n_bad_real = int(bad_real.sum().item())

        if n_bad_real > 0:
            idx = bad_real.nonzero(as_tuple=False).tolist()
            print(f"[Attention Warning] {n_bad_real} REAL rows all -inf; examples (b,h,t): {idx[:5]}")


        # Make empty rows finite (zeros) so softmax is defined; these should correspond to padded queries
        scores = torch.where(row_has_finite, scores, torch.zeros_like(scores))

        # --- Stable softmax ---
        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # keep this for AMP/fp16 safety

        # --- Weighted sum + output projection ---
        attn_output = torch.matmul(attn_weights, V)                 # (B,H,Tq,D)
        out = attn_output.transpose(1, 2).contiguous().view(B, Tq, self.dim)
        out = self.out_proj(out)

        # --- Zero out padded query positions so later temporal blocks can't leak padding back ---
        out = out * q_valid.unsqueeze(-1)  # (B,Tq,dim)

        if self.return_attn:
            return out, attn_weights
        return out



class CausalConv1d(nn.Conv1d):
    """
    1D conv with left-only padding so the output is causal.
    padding = (kernel_size-1)*dilation on the left; no right padding.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        super().__init__(in_ch, out_ch, kernel_size,
                         padding=0, dilation=dilation, bias=bias)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x):  # x: (B, C, T)
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))  # (left, right)
        return super().forward(x)


class TCNBlock(nn.Module):
    """
    Residual TCN block: LN -> GELU -> CausalConv -> Dropout -> LN -> GELU -> CausalConv
    with a (1x1) residual if channels differ.
    """
    def __init__(self, channels, hidden_channels=None, kernel_size=3, dilation=1,
                 dropout=0.1):
        super().__init__()
        hidden = hidden_channels or channels

        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = CausalConv1d(channels, hidden, kernel_size, dilation=dilation, bias=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden)
        self.conv2 = CausalConv1d(hidden, channels, kernel_size, dilation=dilation, bias=True)
        self.drop2 = nn.Dropout(dropout)

        # 1x1 residual if we changed channel dims (here we don’t by default)
        self.res_proj = None
        if channels != channels:
            self.res_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        y = self.norm1(x)
        y = F.gelu(y)
        y = y.transpose(1, 2)                # (B,C,T)
        y = self.conv1(y)                    # (B,hidden,T)
        y = self.drop1(y)
        y = y.transpose(1, 2)                # (B,T,hidden)

        y = self.norm2(y)
        y = F.gelu(y)
        y = y.transpose(1, 2)                # (B,hidden,T)
        y = self.conv2(y)                    # (B,C,T)
        y = self.drop2(y)
        y = y.transpose(1, 2)                # (B,T,C)

        if self.res_proj is not None:
            x_res = self.res_proj(x.transpose(1,2)).transpose(1,2)
        else:
            x_res = x
        return x_res + y


class TCNHead(nn.Module):
    """
    Stack of dilated causal TCN blocks + final linear classifier.
    Keeps (B,T,*) shapes; causal by construction.
    """
    def __init__(self, in_dim=128, hidden_dim=128, num_layers=2,
                 kernel_size=3, dropout=0.1, dilation_base=2,
                 vocab_size=43):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        blocks = []
        for l in range(num_layers):
            dilation = (dilation_base ** l)
            blocks.append(
                TCNBlock(hidden_dim, hidden_channels=hidden_dim,
                         kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    @property
    def receptive_field(self):
        # Rough RF: sum over layers of (k-1)*dilation + 1 (not counting stacking nonlinearities)
        # For num_layers L: RF = 1 + sum_l ( (k-1)*dilation_l )
        return

    def forward(self, x):       # x: (B,T,in_dim)
        y = self.proj_in(x)     # (B,T,H)
        for blk in self.blocks:
            y = blk(y)          # (B,T,H), causal
        y = self.norm_out(y)
        logits = self.classifier(y)  # (B,T,V)
        return logits

    


class CrossAV(nn.Module):
    def __init__(
        self,
        phoneme_vocab_size=43,
        a_drop_modality_prob:float=0.0,
        v_drop_modality_prob:float=0.0,
        return_attn:bool=False,
        # for data augmentation
        use_mfcc_aug: bool = True,
        use_frame_drop: bool = True,
        p_video_frame_drop: float = 0.05,   # fraction of video frames to drop 
        frame_drop_mode: str = "zero",    # "zero" or "prev": replaced by previous frame
        time_mask_p:float=0.06,   # fraction of frames to mask (consecutive frames) 
        freq_mask_p:float=0.1,    # fraction of frequency (Mel) to mask (consecutive bands) 
        n_masks:int=1,          # repeat masking n times, if=2, then 2 regions of frames and 2 regions of frequency will be masked
        # defaults for masking behavior
        fusion_mask_type: str = "none",     # "none" | "causal" | "causal_band"
        fusion_band: int = 0,               # local window radius
        # defaults for Q/K/V projections
        use_q_proj: bool = True,
        use_k_proj: bool = True,
        use_v_proj: bool = True,
        # TCN head config:
        tcn_hidden=128, 
        tcn_layers=2, 
        tcn_kernel=3, 
        tcn_dropout=0.1, 
        tcn_dilation_base=2
    ):
        super().__init__()

        self.audio_encoder  = AuditoryEncoder(in_channels=1)
        self.visual_encoder = VisualEncoder(in_channels=1)

        self.fusion = CrossAttentionFusion(
            dim=128, num_heads=8,
            use_q_proj=use_q_proj,
            use_k_proj=use_k_proj,
            use_v_proj=use_v_proj,
            return_attn=return_attn
        )

        # I used a temporal conv net to causally infer for each fame
        # self.fc = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, phoneme_vocab_size)
        # )

        # swap FC → TCN head (keeps causal)
        self.temporal_head = TCNHead(
            in_dim=128, hidden_dim=tcn_hidden, num_layers=tcn_layers,
            kernel_size=tcn_kernel, dropout=tcn_dropout, dilation_base=tcn_dilation_base,
            vocab_size=phoneme_vocab_size
        )

        self.a_drop_modality_prob = a_drop_modality_prob
        self.v_drop_modality_prob = v_drop_modality_prob
        self.return_attn = return_attn

        # Save fusion defaults
        self.fusion_mask_type = fusion_mask_type
        self.fusion_band = fusion_band

        # ---- store augment config ----
        self.use_mfcc_aug   = use_mfcc_aug
        self.time_mask_p    = float(time_mask_p)
        self.freq_mask_p    = float(freq_mask_p)
        self.n_masks   = int(n_masks)

        self.use_frame_drop = use_frame_drop
        self.p_video_frame_drop   = float(p_video_frame_drop)
        assert frame_drop_mode in ("zero", "prev")
        self.frame_drop_mode = frame_drop_mode

    @staticmethod
    def _build_key_pad_mask(lengths: torch.Tensor, Tk: int):
        """
        Build a right-padding mask for the K/V side.
        lengths: (B,) true (unpadded) lengths
        returns: (B, Tk) bool, True = padding (should be masked)
        """
        device = lengths.device
        B = lengths.shape[0]
        idx = torch.arange(Tk, device=device).unsqueeze(0).expand(B, Tk)  # [0..Tk-1] per row
        return idx >= lengths.unsqueeze(1)
    

    @staticmethod
    # Function to mask out some frquency bands (Mel) and frames in MFCC
    def _specaugment_mfcc(mfcc_btc, time_mask_p: float, freq_mask_p: float, n_masks: int):
        """
        mfcc_btc: (B, T, C) float tensor. Returns augmented copy (B, T, C).
        """
        if (time_mask_p <= 0 and freq_mask_p <= 0) or n_masks <= 0:
            return mfcc_btc
        x = mfcc_btc.clone()
        B, T, C = x.shape
        for b in range(B):
            for _ in range(n_masks):
                # time mask
                t = int(T * time_mask_p)
                if t > 0:
                    t0 = torch.randint(low=0, high=max(1, T - t + 1), size=(1,), device=x.device).item()
                    x[b, t0:t0 + t, :] = 0
                # freq mask
                f = int(C * freq_mask_p)
                if f > 0:
                    f0 = torch.randint(low=0, high=max(1, C - f + 1), size=(1,), device=x.device).item()
                    x[b, :, f0:f0 + f] = 0
        return x

    @staticmethod
    # Function to drop frames in videos
    def _random_frame_drop(video_bthw, p_video_frame_drop: float, mode: str = "zero"):
        """
        video_bthw: (B, T, H, W) float in [0,1]. Returns augmented copy.
        mode="zero": set dropped frames to 0; mode="prev": copy previous frame (if t>0).
        """
        if p_video_frame_drop <= 0:
            return video_bthw
        x = video_bthw.clone()
        B, T, H, W = x.shape
        drop_mask = torch.rand(B, T, device=x.device) < p_video_frame_drop  # True => drop
        if mode == "zero":
            x[drop_mask] = 0.0
        else:  # "prev"
            # for t==0, simply zero (or leave as is)
            for b in range(B):
                for t in range(T):
                    if drop_mask[b, t]:
                        if t > 0:
                            x[b, t] = x[b, t - 1]
                        else:
                            x[b, t] = 0.0
        return x

  

    def forward(self, audio_tensor=None, visual_tensor=None, mode="av",
                lengths_audio=None, lengths_video=None, attn_mask=None):
        """
        audio_tensor:  (B, T_a, n_mfcc)  float
        visual_tensor: (B, T_v, H, W)    float
        """
        if audio_tensor is None and visual_tensor is None:
            raise ValueError("No input modality provided.")

        # Modality dropping (training-time only; optional)
        if self.training and mode == "av":
            a_dropped = torch.rand(1).item() < self.a_drop_modality_prob
            v_dropped = torch.rand(1).item() < self.v_drop_modality_prob
            if   a_dropped and not v_dropped: mode = "v"
            elif v_dropped and not a_dropped: mode = "a"
            elif a_dropped and v_dropped:     mode = "a" if torch.rand(1).item() < 0.5 else "v"

        # Augment input data
        # Apply MFCC SpecAugment on (B,T,n_mfcc) directly, masking frames and/or mel bands
        if self.training and (mode in ["av", "a"]) and (audio_tensor is not None) and self.use_mfcc_aug:
            audio_tensor = self._specaugment_mfcc(
                audio_tensor, self.time_mask_p, self.freq_mask_p, self.n_masks
            )

        # Apply frame drop on raw frames (B,T,H,W) before visual encoder
        if self.training and (mode in ["av", "v"]) and (visual_tensor is not None) and self.use_frame_drop:
            visual_tensor = self._random_frame_drop(
                visual_tensor, self.p_video_frame_drop, mode=self.frame_drop_mode
            )

        # --- Encode audio ---
        audio_feat = None
        if (mode in ["av", "a"]) and (audio_tensor is not None):
            # (B, T, n_mfcc) -> (B, 1, n_mfcc, T)
            x_a = audio_tensor.permute(0, 2, 1).unsqueeze(1).contiguous().float()
            audio_feat = self.audio_encoder(x_a)  # (B, T_a, 128)

        # --- Encode video ---
        visual_feat = None
        if (mode in ["av", "v"]) and (visual_tensor is not None):
            x_v = visual_tensor.float()           # (B, T_v, H, W)
            visual_feat = self.visual_encoder(x_v)  # (B, T_v, 128)

        # Sanity checks for unimodal modes
        if mode == "a" and audio_feat is None:
            raise ValueError("Mode 'a' requires audio_tensor.")
        if mode == "v" and visual_feat is None:
            raise ValueError("Mode 'v' requires visual_tensor.")

        attn_w = None  # will be filled if return_attn=True

        # --- Time alignment before fusion ---
        if mode == "av":
            Ta = audio_feat.shape[1]
            Tv = visual_feat.shape[1]
            assert Ta == Tv, f"Time length mismatch: audio={Ta}, video={Tv}"
            # Build K/V padding mask from audio lengths (since K,V = audio_feat)
            key_padding_mask = None
            if lengths_audio is not None:
                key_padding_mask = self._build_key_pad_mask(lengths_audio.to(audio_feat.device), Tk=Ta)

            # --- Fusion (Q=visual, K/V=audio) ---
            if self.fusion.return_attn:
                fused, attn_w = self.fusion(
                    visual_feat, audio_feat, audio_feat,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    mask_type=self.fusion_mask_type,
                    band=self.fusion_band
                )
            else:
                fused = self.fusion(
                    visual_feat, audio_feat, audio_feat,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    mask_type=self.fusion_mask_type,
                    band=self.fusion_band
                )

        elif mode == "a":
            Ta = audio_feat.shape[1]
            key_padding_mask = None
            if lengths_audio is not None:
                key_padding_mask = self._build_key_pad_mask(lengths_audio.to(audio_feat.device), Tk=Ta)
            if self.fusion.return_attn:
                fused, attn_w = self.fusion(
                    audio_feat, audio_feat, audio_feat,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    mask_type=self.fusion_mask_type,
                    band=self.fusion_band
                )
            else:
                fused = self.fusion(
                    audio_feat, audio_feat, audio_feat,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    mask_type=self.fusion_mask_type,
                    band=self.fusion_band
                )

        else:  # mode == "v"
            Tv = visual_feat.shape[1]
            key_padding_mask = None
            if lengths_video is not None:
                key_padding_mask = self._build_key_pad_mask(lengths_video.to(visual_feat.device), Tk=Tv)
            if self.fusion.return_attn:
                fused, attn_w = self.fusion(
                    visual_feat, visual_feat, visual_feat,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    mask_type=self.fusion_mask_type,
                    band=self.fusion_band
                )
            else:
                fused = self.fusion(
                    visual_feat, visual_feat, visual_feat,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    mask_type=self.fusion_mask_type,
                    band=self.fusion_band
                )

        # --- Classifier ---
        
        #logits = self.fc(fused)
        logits = self.temporal_head(fused)

        if self.fusion.return_attn:
            return logits, attn_w
        return logits




































##### old codes #####



class CrossAttentionFusion_old(nn.Module):
    def __init__(self, dim, num_heads=8, return_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.return_attn = return_attn

    def forward(self, query, key, value, attn_mask=None):
        B, Tq, _ = query.shape
        Bk, Tk, _ = key.shape
        assert B == Bk, "Batch size mismatch between Q and K/V"

        Q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tq, D)
        K = self.k_proj(key).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tk, D)
        V = self.v_proj(value).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tk, D)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)       # (B, H, Tq, Tk)
        if attn_mask is not None:
            # attn_mask should be broadcastable to (B, H, Tq, Tk). 1=keep, 0=mask.
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)                                  # (B, H, Tq, Tk)
        attn_output  = torch.matmul(attn_weights, V)                                       # (B, H, Tq, D)

        out = attn_output.transpose(1, 2).contiguous().view(B, Tq, self.dim)               # (B, Tq, dim)
        out = self.out_proj(out)

        if self.return_attn:
            return out, attn_weights
        return out



class CrossAV_old(nn.Module):
    def __init__(self, phoneme_vocab_size=42, a_drop_modality_prob=0.0, v_drop_modality_prob=0.0,
                 return_attn=False):
        super().__init__()

        # Expect MFCC as (B, T, n_mfcc) from collate
        self.audio_encoder  = AuditoryEncoder(in_channels=1)   # we'll feed (B,1,n_mfcc,T)
        self.visual_encoder = VisualEncoder(in_channels=1)     # expects (B, T, H, W)

        self.fusion = CrossAttentionFusion(dim=128, num_heads=8, use_k_proj=True,
                                           use_q_proj=True, use_v_proj=True, return_attn=return_attn)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, phoneme_vocab_size)
        )

        self.a_drop_modality_prob = a_drop_modality_prob
        self.v_drop_modality_prob = v_drop_modality_prob
        self.return_attn = return_attn

    def forward(self, audio_tensor=None, visual_tensor=None, mode="av"):
        """
        audio_tensor:  (B, T_a, n_mfcc)  float
        visual_tensor: (B, T_v, H, W)    float
        """
        if audio_tensor is None and visual_tensor is None:
            raise ValueError("No input modality provided.")

        # Modality dropping (training-time only; optional)
        if self.training and mode == "av":
            a_dropped = torch.rand(1).item() < self.a_drop_modality_prob
            v_dropped = torch.rand(1).item() < self.v_drop_modality_prob
            if   a_dropped and not v_dropped: mode = "v"
            elif v_dropped and not a_dropped: mode = "a"
            elif a_dropped and v_dropped:     mode = "a" if torch.rand(1).item() < 0.5 else "v" # pick mode again if both dropped

        B = (audio_tensor.shape[0] if audio_tensor is not None else visual_tensor.shape[0])

        # --- Encode audio ---
        audio_feat = None
        if (mode in ["av", "a"]) and (audio_tensor is not None):
            # (B, T, n_mfcc) -> (B, 1, n_mfcc, T)
            x_a = audio_tensor.permute(0, 2, 1).unsqueeze(1).contiguous().float()
            audio_feat = self.audio_encoder(x_a)  # (B, T_a, 128)

        # --- Encode video ---
        visual_feat = None
        if (mode in ["av", "v"]) and (visual_tensor is not None):
            x_v = visual_tensor.float()           # (B, T_v, H, W)
            visual_feat = self.visual_encoder(x_v)  # (B, T_v, 128)

        # Sanity checks for unimodal modes
        if mode == "a" and audio_feat is None:
            raise ValueError("Mode 'a' requires audio_tensor.")
        if mode == "v" and visual_feat is None:
            raise ValueError("Mode 'v' requires visual_tensor.")


        # --- Time alignment before fusion ---
        if mode == "av":
            Ta = audio_feat.shape[1]
            Tv = visual_feat.shape[1]
            assert Ta == Tv, f"Time length mismatch: audio={Ta}, video={Tv}"
            T = Ta  # or Tv, since they're equal
        else:
            T = (audio_feat.shape[1] if mode == "a" else visual_feat.shape[1]) # just to get a T for consistency, not really necesary


        # # --- Time alignment before fusion ---
        # if mode == "av":
        #     Ta = audio_feat.shape[1]
        #     Tv = visual_feat.shape[1]
        #     T = min(Ta, Tv)
        #     if Ta != Tv:
        #         # Trim both to the common T
        #         audio_feat  = audio_feat[:,  :T, :]
        #         visual_feat = visual_feat[:, :T, :]
        # else:
        #     # Unimodal: pick T from available feat
        #     T = (audio_feat.shape[1] if mode == "a" else visual_feat.shape[1])

        # --- Fusion ---
        if self.fusion.return_attn:
            if mode == "av":
                fused, attn_w = self.fusion(visual_feat, audio_feat, audio_feat)      # (B, T, 128), (B,H,T,T)
            elif mode == "a":
                fused, attn_w = self.fusion(audio_feat, audio_feat, audio_feat)
            else:  # mode == "v"
                fused, attn_w = self.fusion(visual_feat, visual_feat, visual_feat)
        else:
            if mode == "av":
                fused = self.fusion(visual_feat, audio_feat, audio_feat)              # (B, T, 128)
            elif mode == "a":
                fused = self.fusion(audio_feat, audio_feat, audio_feat)
            else:  # mode == "v"
                fused = self.fusion(visual_feat, visual_feat, visual_feat)

        # --- Classifier ---
        logits = self.fc(fused)  # (B, T, vocab)

        if self.fusion.return_attn:
            return logits, attn_w
        return logits
