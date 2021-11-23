import torch
import torch.nn as nn
from models.MIC_transformer import TransformerModel
from models.MIC_transformer.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from models.MIC_transformer.Unet_skipconnection import Unet


class MIC_Transformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(MIC_Transformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:

            self.conv_x = nn.Conv3d(
                128,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.Unet = Unet(in_channels=1, base_channels=8, num_classes=3)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(1728*128,1024),
        nn.ReLU(True),

        nn.Dropout(),
        nn.Linear(1024,128),
        nn.Sigmoid(),

        nn.Linear(128,2),
        )


    def encode(self, x):
        # combine embedding with conv patch distribution
        x0_1, x1_1, x2_1, x3_1, x = self.Unet(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x0_1, x1_1, x2_1, x3_1, x, intmd_x



    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):

        x0_1, x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs = self.encode(x)

        x = torch.flatten(encoder_output)

        return self.classifier(x)

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

class MIC(MIC_Transformer):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(BTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

def MIC_Model(dataset='mri', _conv_repr=True, _pe_type="learned"):

    if dataset.lower() == 'mri':
        img_dim = 192
        num_classes = 3

    num_channels = 1
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = MIC(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        hidden_dim=1728,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model
