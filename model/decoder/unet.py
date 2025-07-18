from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.diffusion_utils import timestep_embedding, ResBlock, SpatialTransformer
from model.backbone import build_backbone

class UNetModel(nn.Module):
    def __init__(self, cfg) -> None:
        super(UNetModel, self).__init__()

        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x
        self.plus_condition_type = cfg.plus_condition_type
        self.trans_condition_type = cfg.trans_condition_type

        ## create scene model from config
        self.scene_model = build_backbone(cfg.backbone)
        if hasattr(cfg, "use_hand") and cfg.use_hand:
            self.hand_model = build_backbone(cfg.hand_backbone)
        if hasattr(cfg, "use_guidance") and cfg.use_guidance:
            self.language_model = build_backbone(cfg.language_encoder)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.plus_condition_type,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model,
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
    def condition_mask(self, obj_cond, txt_cond, text_vector):
        cond = torch.cat([obj_cond, txt_cond], dim=1)
        batch_size, seq_length, embedding_size = cond.size()

        txt_index = text_vector.argmax(dim=-1) + cond.shape[0] + 2

        range_matrix = torch.arange(seq_length).expand(batch_size, seq_length).to(obj_cond.device)

        attention_mask = range_matrix < txt_index.unsqueeze(1)
        attention_mask[:,cond.shape[0]+1] = False
        return cond, attention_mask
    
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, data:Dict) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """

        if self.trans_condition_type == "txt_obj":
            cond = torch.cat([data["cond_obj"], data["cond_txt"]], dim=1)
            atten_mask = None 
        elif self.trans_condition_type == "obj_hand":
            cond = torch.cat([data["cond_obj"], data["cond_hand"]], dim=1)
            atten_mask = None
        else:
            raise Exception("no valid trans condition")
        
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## time embedding
        if ts != None:
            t_emb = timestep_embedding(ts, self.d_model)
            t_emb = self.time_embed(t_emb)
        else:
            t_emb = None

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb, data)
            h = self.layers[i * 2 + 1](h, context=cond, mask=atten_mask)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h 

    def condition_obj(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """

        b = data['obj_pc'].shape[0]
        obj_pc = data['obj_pc'].to(torch.float32)
        _, obj_feat = self.scene_model(obj_pc)
        obj_feat = obj_feat.permute(0,2,1).contiguous()
        # (B, C, N)
        return obj_feat

    def condition_language(self, data: Dict) -> torch.Tensor:

        cond_txt, text_vector = self.language_model(data["guidance"], data["clip_data"] if "clip_data" in data else None)
        cond_txt_cls = cond_txt[torch.arange(cond_txt.shape[0]), text_vector.argmax(dim=-1)]

        return cond_txt_cls, cond_txt

    def condition_hand(self, data: Dict) -> torch.Tensor:
        b = data['hand_pc'].shape[0]
        hand_pc = data['hand_pc']
        _, hand_feat = self.hand_model(hand_pc)
        hand_feat = hand_feat.permute(0,2,1).contiguous()
        # (B, C, N)
        return hand_feat

