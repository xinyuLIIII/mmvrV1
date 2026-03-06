import torch
import torch.nn.functional as F
from torch import nn

from models.Transformer_layers import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerPoseDecoder,
    TransformerPoseDecoderLayer,
)
from models.backbone import build_backbone
from models.position_encoding import build_position_encoding
from utils.loss import SetCriterion
from utils.matcher import build_matcher

try:
    from mamba_ssm import Mamba
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc


class MissingMambaDependencyError(ImportError):
    pass


def _raise_mamba_dependency_error():
    message = (
        "mmVR_Mamba requires the official `mamba-ssm` implementation and its "
        "`causal-conv1d` dependency. Install them in a dedicated environment "
        "(recommended: Python >= 3.8, torch >= 1.12, CUDA >= 11.6) with "
        "`pip install causal-conv1d>=1.4.0 mamba-ssm`."
    )
    raise MissingMambaDependencyError(message) from _MAMBA_IMPORT_ERROR


class prediction_head(nn.Module):
    def __init__(self, hidden_dim, num_logitsclasses, num_idclasses):
        super().__init__()
        self.logitsclass_embed = nn.Linear(hidden_dim, num_logitsclasses + 1)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 21 * 3, 3)

    def forward(self, tgt):
        outputs_logi = self.logitsclass_embed(tgt)
        outputs_kpt = self.kpt_embed(tgt)
        return outputs_logi, outputs_kpt


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        if Mamba is None:
            _raise_mamba_dependency_error()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = self.dropout(x)
        return residual + x


class MambaTemporalDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_frames,
        num_queries,
        num_layers=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.temporal_blocks = nn.ModuleList(
            ResidualMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.query_blocks = nn.ModuleList(
            ResidualMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.context_gate = nn.Linear(d_model * 2, d_model)

    def forward(self, tgt, pose_memory, memory, pos=None, query_pos=None):
        query = tgt.transpose(0, 1)
        temporal = (pose_memory + memory).transpose(0, 1)

        if pos is not None:
            temporal = temporal + pos.transpose(0, 1)
        if query_pos is not None:
            query = query + query_pos.transpose(0, 1)

        query = self.query_norm(query)
        for block in self.temporal_blocks:
            temporal = block(temporal)
        temporal = self.temporal_norm(temporal)

        batch_size = query.shape[0]
        frame_queries = query.unsqueeze(1).expand(-1, self.num_frames, -1, -1)
        frame_context = temporal.unsqueeze(2).expand(-1, -1, self.num_queries, -1)
        gate = torch.sigmoid(self.context_gate(torch.cat([frame_queries, frame_context], dim=-1)))
        fused = self.fusion_norm(frame_queries + gate * frame_context)
        fused = fused.reshape(batch_size * self.num_frames, self.num_queries, -1)

        for block in self.query_blocks:
            fused = block(fused)

        fused = fused.reshape(batch_size, self.num_frames, self.num_queries, -1)
        return fused.permute(1, 2, 0, 3).contiguous()


class mmVR_Mamba(nn.Module):
    def __init__(
        self,
        args,
        backbone,
        backbone_name,
        d_model=512,
        nhead=8,
        num_frames=30,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_logitsclass=1,
        num_queries=100,
        return_intermediate_dec=False,
        num_class=8,
        mamba_layers=6,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    ):
        super().__init__()
        self.hidden_dim = d_model
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.mmwave_backbone = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.fc_mmwave = nn.Linear(64 * 8 * 4, 512)
        self.imu_backbone = nn.LSTM(input_size=6, hidden_size=512, num_layers=2, batch_first=True)
        self.position_embedding = build_position_encoding(args)
        self.num_channels = 512 if backbone_name in ('resnet18', 'resnet34') else 2048
        self.mem_stats = getattr(args, 'mem_stats', False)

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        encoder_layer_imu = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm_imu = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_imu = TransformerEncoder(encoder_layer_imu, num_encoder_layers, encoder_norm_imu)

        posedecoder_layer = TransformerPoseDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        posedecoder_norm = nn.LayerNorm(d_model)
        self.posedecoder = TransformerPoseDecoder(
            posedecoder_layer,
            num_decoder_layers,
            posedecoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.temporaldecoder = MambaTemporalDecoder(
            d_model=d_model,
            num_frames=num_frames,
            num_queries=num_queries,
            num_layers=mamba_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
        )
        self.posequery_embed = nn.Embedding(num_frames, self.hidden_dim)
        self.temporalquery_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.prediction_head = prediction_head(self.hidden_dim, num_logitsclass, num_class)
        self.input_proj = nn.Conv2d(self.num_channels, self.hidden_dim, kernel_size=1)
        self.feature_linear = nn.Linear(256 * 128, self.hidden_dim)
        self.feature_linear_imu = nn.Linear(6, self.hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, src, imu):
        src = src.unsqueeze(1)
        batch_size = src.shape[0]

        src = self.mmwave_backbone(src)
        src = src.permute(0, 2, 1, 3, 4).contiguous()
        src = src.view(src.shape[0], src.shape[1], -1)
        src = self.fc_mmwave(src)

        imu, _ = self.imu_backbone(imu)
        pos_embed_mmwave = self.position_embedding(src.reshape(batch_size, self.num_frames, 32, 16))
        pos_embed_mmwave = pos_embed_mmwave.flatten(2).permute(1, 0, 2)
        pos_embed_imu = self.position_embedding(imu.reshape(batch_size, self.num_frames, 32, 16))
        pos_embed_imu = pos_embed_imu.flatten(2).permute(1, 0, 2)
        pos_embed = 0.9 * pos_embed_mmwave + 0.1 * pos_embed_imu

        src = src.permute(1, 0, 2)
        imu = imu.permute(1, 0, 2)

        memory_imu = self.encoder_imu(imu, pos=pos_embed_imu)
        memory = self.encoder(src, pos=pos_embed_mmwave)
        memory = 0.1 * memory_imu + 0.9 * memory

        posequery_embed = self.posequery_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        temporalquery_embed = self.temporalquery_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt_pose = torch.zeros_like(posequery_embed)
        tgt_tem = torch.zeros_like(temporalquery_embed)

        hs = self.posedecoder(tgt_pose, memory, pos=pos_embed, query_pos=posequery_embed)
        pose_memory = hs[-1, :, :, :]
        temporal_out = self.temporaldecoder(
            tgt_tem,
            pose_memory,
            memory,
            pos=pos_embed,
            query_pos=temporalquery_embed,
        )
        temporal_out = temporal_out.transpose(1, 2)
        outputs_logi, outputs_kpt = self.prediction_head(temporal_out)
        out = {'pred_logits': outputs_logi, 'pred_kpt': outputs_kpt}
        if self.mem_stats:
            out['pose_memory'] = pose_memory.detach()
        return out


def build_model(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    model = mmVR_Mamba(
        args,
        backbone,
        args.backbone_name,
        d_model=512,
        nhead=args.nheads,
        num_frames=30,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation='relu',
        normalize_before=True,
        num_logitsclass=1,
        num_queries=args.num_queries,
        return_intermediate_dec=True,
        num_class=8,
        mamba_layers=args.mamba_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_kpt': args.kpt_loss_coef}
    losses = ['cls', 'kpt']
    criterion = SetCriterion(
        num_logitsclass=1,
        matcher=matcher,
        weight_dict=weight_dict,
        logits_eos_coef=args.cls_loss_coef,
        losses=losses,
    )
    criterion.to(device)
    return model, criterion
