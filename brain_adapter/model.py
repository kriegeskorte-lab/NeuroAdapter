import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math

from brain_adapter.transformer import Transformer

'''
ParcelMapper: Map raw fMRI data to fixed-dim tokens per parcel.
'''
class ParcelMapper(nn.Module):
    def __init__(self, num_parcels, max_voxels, out_dim=768):
        """
        Linear projection of brain data to match the image embedding dimension.
        Projects [B, num_parcels, max_voxels] → [B, num_parcels, out_dim]
        """
        super().__init__()
        self.linear_weights = nn.Parameter(torch.empty(num_parcels, max_voxels, out_dim))
        self.linear_bias = nn.Parameter(torch.empty(num_parcels, out_dim))
        self.num_parcels = num_parcels
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_weights)
        nn.init.zeros_(self.linear_bias)

    def forward(self, x):
        # x: [B, P, V], linear_weights: [parPcels, V, 768], linear_bias: [P, 768]
        # print(f"Input shape: {x.shape}, Linear weights shape: {self.linear_weights.shape}, Linear bias shape: {self.linear_bias.shape}")
        return torch.einsum('bpv,pvd->bpd', x, self.linear_weights) + self.linear_bias

'''
Transformer Decoder Architecture
'''
# class TokenMapper(nn.Module):
#     def __init__(self, 
#                  num_parcels=200,
#                  num_decoder_queries=50, 
#                  d_model=768,
#                  num_decoder_layers=1,
#                  nhead=8,
#                  dropout=0.1):
#         super().__init__()
        
#         # Learnable queries for the decoder
#         self.decoder_queries = nn.Embedding(num_decoder_queries, d_model) 
#         self.roi_embeddings = nn.Embedding(num_parcels, d_model) 

#         # Only transformer decoder: queries attend to fMRI tokens
#         decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

#     def forward(self, fmri_tokens):
#         """
#         fmri_tokens: [B, num_fmri_tokens, d_model] 
#         returns: [B, num_decoder_queries, d_model]
#         """  
#         B, num_parcels, d_model = fmri_tokens.shape
#         num_queries = self.decoder_queries.num_embeddings

#         pos_embeddings = self.roi_embeddings.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_parcels, d_model]
#         keys = fmri_tokens + pos_embeddings  # [B, d_model, num_parcels]
#         queries = self.decoder_queries.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, d_model]

#         condition_tokens = self.decoder(
#             tgt=queries,
#             memory=keys,
#         ) # [B, num_queries, d_model]

#         return condition_tokens
    
class TokenMapper(nn.Module):
    def __init__(self, 
                 num_parcels=200,
                 num_decoder_queries=50, 
                 d_model=768,
                 num_decoder_layers=1,
                 nhead=8,
                 dropout=0.1):
        super().__init__()
        
        # Learnable queries for the decoder
        self.decoder_queries = nn.Embedding(num_decoder_queries, d_model) 
        self.roi_embeddings = nn.Embedding(num_parcels, d_model) 

        self.transformer = Transformer(
            d_model=d_model, 
            dropout=dropout, 
            nhead=nhead, 
            dim_feedforward = 1024,
            num_encoder_layers=0, 
            num_decoder_layers=num_decoder_layers,
            normalize_before=True, 
            return_intermediate_enc=False,
            return_intermediate_dec=False,
            enc_output_layer=1,
        )

    def forward(self, fmri_tokens):
        """
        fmri_tokens: [B, num_fmri_tokens, d_model] 
        returns: [B, num_decoder_queries, d_model]
        """  
        B, num_parcels, d_model = fmri_tokens.shape
        num_queries = self.decoder_queries.num_embeddings

        # [B, num_parcels, d_model] → [B, d_model, num_parcels, 1]
        src = fmri_tokens.permute(0, 2, 1).unsqueeze(-1)   # [B, d_model, num_parcels, 1]

        # [num_parcels, d_model] → [1, num_parcels, d_model] → [B, num_parcels, d_model] → [B, d_model, num_parcels]
        pos_embed = self.roi_embeddings.weight.unsqueeze(0).repeat(B, 1, 1).permute(0, 2, 1).unsqueeze(-1)  # [B, d_model, num_parcels, 1]

        # [num_queries, d_model]
        query_embed = self.decoder_queries.weight    # [num_queries, d_model]
        mask = torch.zeros(B, num_parcels, device=fmri_tokens.device)  # [B, num_parcels]

        # Now call transformer exactly as before
        condition_tokens = self.transformer.forward(
            src=src,
            mask=mask,
            query_embed=query_embed,
            pos_embed=pos_embed,
        ).squeeze(0)  # [B, num_decoder_queries, d_model]

        return condition_tokens
    
'''
GuidanceGenerator: Full pipeline, fMRI to condition tokens
'''
class GuidanceGenerator(nn.Module):
    def __init__(self, num_parcels=200, max_voxels=564, output_dim=768, num_decoder_queries=50,
                 num_decoder_layers=1, nhead=8, dropout=0.1, sub_approach='transformer_decoder'):
        """
        Combines ParcelMapper and decoder-only TokenMapper.
        """
        super().__init__()
        self.parcel_mapper = ParcelMapper(num_parcels=num_parcels, max_voxels=max_voxels, out_dim=output_dim)
        self.sub_approach = sub_approach
        if self.sub_approach == 'transformer_decoder':
            self.token_mapper = TokenMapper(
                num_parcels=num_parcels,
                num_decoder_queries=num_decoder_queries,
                d_model=output_dim,
                num_decoder_layers=num_decoder_layers,
                nhead=nhead,
                dropout=dropout
            )
        
    def forward(self, fmri_data):
        """
        fmri_data: [B, num_parcels, max_voxels]
        Returns:
            condition_tokens: [B, num_decoder_queries, output_dim]
            fmri_tokens: [B, num_parcels, output_dim]
        """
        fmri_tokens = self.parcel_mapper(fmri_data)
        # print(torch.isnan(self.parcel_mapper.linear_weights).any(), 'NaNs in weights')
        # print(torch.isnan(self.parcel_mapper.linear_bias).any(), 'NaNs in bias')
        # print(torch.isnan(fmri_tokens).any(), 'NaNs in fmri_tokens!')
        # print(torch.isnan(fmri_data).any(), 'NaNs in fmri_data!')
        # return fmri_tokens, None

        if self.sub_approach == 'transformer_decoder':
            condition_tokens = self.token_mapper(fmri_tokens)
            return condition_tokens, fmri_tokens
        else:
            return fmri_tokens, None  # We use fmri tokens from parcel-wise linear mapping as condition tokens directly

class NeuroAdapter(torch.nn.Module):
    """
    NeuroAdapter: Conditions Stable Diffusion’s U-Net on fMRI-derived tokens.

    This adapter takes raw fMRI → condition tokens (via a separate GuidanceGenerator),
    projects them into the U-Net’s cross-attention space, and uses them in place of text.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        image_proj_model: torch.nn.Module,
        adapter_modules: torch.nn.Module,
        ckpt_path: str = None
    ):
        """
        Args:
            unet:              The diffusion U-Net (from diffusers.UNet2DConditionModel).
            image_proj_model:  Projects condition_tokens (B×N×768) → (B×N×cross_attn_dim).
            adapter_modules:   The IPAttnProcessor modules (to_k_ip/to_v_ip layers).
            ckpt_path:         Optional path to load pretrained adapter weights.
        """
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        condition_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one denoising step, conditioning on fMRI tokens.

        Args:
            noisy_latents:        [B, C, H, W] noised image latents.
            timesteps:            [B] current diffusion timesteps.
            encoder_hidden_states:[B, L_text, D_text] text embeddings (can be empty).
            condition_tokens:     [B, N, 768] fMRI-derived tokens.

        Returns:
            noise_pred:           [B, C, H, W] predicted noise residual.
        """
        # 1) Project fMRI tokens into U-Net’s attention key/value space:
        #    ip_tokens.shape == [B, N, cross_attention_dim]
        ip_tokens = self.image_proj_model(condition_tokens)

        # 2) Concatenate with (possibly empty) text embeddings along token axis:
        #    combined.shape == [B, L_text + N, D_text]
        encoder_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)

        # 3) Run the U-Net denoiser with our combined context:
        #    .sample is the predicted noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_states).sample

        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        """
        Load pretrained adapter weights and verify they’ve changed.
        Expects 'image_proj' and 'ip_adapter' keys in checkpoint.
        """
        # Record sums before loading
        orig_proj_sum = sum(torch.sum(p) for p in self.image_proj_model.parameters())
        orig_mod_sum  = sum(torch.sum(p) for p in self.adapter_modules.parameters())

        ck = torch.load(ckpt_path, map_location='cpu')
        self.image_proj_model.load_state_dict(ck['image_proj'], strict=True)
        self.adapter_modules.load_state_dict(ck['ip_adapter'], strict=True)

        # Verify sums changed
        new_proj_sum = sum(torch.sum(p) for p in self.image_proj_model.parameters())
        new_mod_sum  = sum(torch.sum(p) for p in self.adapter_modules.parameters())
        assert orig_proj_sum != new_proj_sum, "Projection weights did not change!"
        assert orig_mod_sum  != new_mod_sum,  "Adapter module weights did not change!"

        print(f"Loaded NeuroAdapter checkpoint from {ckpt_path}")

# ------------------------------
# Testing with Fake Data
# ------------------------------
if __name__ == "__main__":

    B, num_parcels, max_voxels = 2, 100, 564*2
    fmri_data = torch.randn(B, num_parcels, max_voxels, requires_grad=True)
    print("Simulated fMRI data shape:", fmri_data.shape)  # Expected: [2, 100, 1128]
    
    # Initialize the autoencoder with token dimension of 768.
    model = GuidanceGenerator(num_parcels=num_parcels, max_voxels=max_voxels, num_decoder_queries=50, output_dim=768)
    condition_token, fmri_features = model(fmri_data)

    print("Output fMRI features shape:", fmri_features.shape)  # Expected: [2, 1000, 768]
    print("Output condition token shape:", condition_token.shape)  # Expected: [2, 961, 768]
    
    loss = condition_token.norm()
    loss.backward()

    # Now check
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
    print("Grad checked")