import os
from timm.models.helpers import load_state_dict
from timm.models.layers import set_layer_config
import torch.nn as nn
import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Parameter, init, LSTM
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from .utils.tokenizer import Tokenizer
from .utils.transformers import TransformerEncoderLayer
from .utils.helpers import pe_check

try:
    from timm.models.registry import register_model, model_entrypoint
except ImportError:
    pass

model_urls = {
    'cct_14_7x2_224':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
}

class SpeakerVGGVoxCCTEndToEnd(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 type='pooling',  # pooling | speaker_token
                 gender_embedder=None,
                 corpus_embedder=None,
                 *args, **kwargs):
        super(SpeakerVGGVoxCCTEndToEnd, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)
        self.classifier = SpeakerVGGVoxCCTEndToEndClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size) + 1,
            embedding_dim=embedding_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            gender_embedder=gender_embedder,
            corpus_embedder=corpus_embedder,
            type=type,
        )


    def forward(self, x):
        x_spectogram = x[0]
        x_speaker = x[1]
        x_emotion = self.tokenizer(x_spectogram)
        return self.classifier(x_emotion, x_speaker, x_spectogram)

    def weight_regularization(self):
        return self.classifier.weight_regularization()


class SpeakerVGGVoxCCTEndToEndClassifier(Module):
    def __init__(self,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 type='pooling',
                 gender_embedder=None,
                 corpus_embedder=None,
                 ):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_tokens = 0
        self.num_classes = num_classes
        self.type = type

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if type == 'pooling':
            self.attention_pool_cct = Linear(self.embedding_dim, 1)

        self.gender_embedder = gender_embedder
        self.corpus_embedder = corpus_embedder

        if gender_embedder is not None:
            sequence_length += 1
        if corpus_embedder is not None:
            sequence_length += 1

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])

        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)

        self.apply(self.init_weight)

    def forward(self, x_emotion, x_speaker, x_spectrogram):
        x = x_emotion
        if self.gender_embedder:
            x_gender = self.gender_embedder.extract_embedding(x_spectrogram)
            x = torch.cat((x_gender.unsqueeze(1), x), dim=1)
        if self.corpus_embedder:
            x_corpus = self.corpus_embedder.extract_embedding(x_spectrogram)
            x = torch.cat((x_corpus.unsqueeze(1), x), dim=1)

        x = torch.cat((x_speaker.unsqueeze(1), x), dim=1)

        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.type == 'pooling':
            x = torch.matmul(F.softmax(self.attention_pool_cct(x), dim=1).transpose(-1, -2),
                                     x).squeeze(-2)
        else:
            x = x[:, 0, :]

        x = self.fc(x)

        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def weight_regularization(self):
        return torch.norm(self.fc.weight, p=1)

def _read_embedder_model(embedder_path, num_embedder_classes=2):
    model_name = 'cct_14_7x2_224'
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(num_classes=num_embedder_classes)

    if os.path.splitext(embedder_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(embedder_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    embedder_path = os.path.abspath(embedder_path)
    state_dict = load_state_dict(embedder_path, False)
    model.load_state_dict(state_dict, strict=True)
    return model

def _speaker_vgg_cct_end_to_end(pretrained, progress, pretrained_arch,
                     num_layers, num_heads, mlp_ratio, embedding_dim,
                     gender=False, gender_embedder_path=None,
                     corpus=False, corpus_embedder_path=None,
                     kernel_size=3, stride=None, padding=None, positional_embedding='learnable',
                     *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    gender_embedder = None
    if gender:
        gender_embedder = _read_embedder_model(gender_embedder_path, num_embedder_classes=2)
    corpus_embedder = None
    if corpus:
        corpus_embedder = _read_embedder_model(corpus_embedder_path, num_embedder_classes=6)
    model = SpeakerVGGVoxCCTEndToEnd(num_layers=num_layers,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             embedding_dim=embedding_dim,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             gender_embedder=gender_embedder,
                             corpus_embedder=corpus_embedder,
                             *args, **kwargs)

    if pretrained:
        if pretrained_arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[pretrained_arch],
                                                  progress=progress)
            if positional_embedding == 'learnable':
                state_dict = pe_check(model, state_dict)
            elif positional_embedding == 'sine':
                state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']

            model_dict = model.state_dict()

            state_dict = {k: v for k, v in state_dict.items() if
                          k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError(f'Variant {pretrained_arch} does not yet have pretrained weights.')

    return model

def speaker_vgg_cct_end_to_end_14(pretrained, progress, *args, **kwargs):
    return _speaker_vgg_cct_end_to_end(pretrained=pretrained, progress=progress,
                            num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                            *args, **kwargs)

@register_model
def speaker_vgg_cct_end_to_end_14_7x2_224(pretrained=False, progress=False, img_size=224,
                                          positional_embedding='learnable', num_classes=10, *args, **kwargs):
    return speaker_vgg_cct_end_to_end_14(arch='speaker_vgg_cct_end_to_end_14_7x2_224',
                                         pretrained=pretrained, progress=progress,
                             pretrained_arch="cct_14_7x2_224",
                             kernel_size=7, n_conv_layers=2,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)

@register_model
def speaker_vgg_cct_end_to_end_speaker_token_14_7x2_224(pretrained=False, progress=False, img_size=224,
                                                        positional_embedding='learnable', num_classes=10,
                                                        *args, **kwargs):
    return speaker_vgg_cct_end_to_end_14(arch='speaker_vgg_cct_end_to_end_speaker_token_14_7x2_224',
                                         pretrained=pretrained, progress=progress,
                             pretrained_arch="cct_14_7x2_224",
                             kernel_size=7, n_conv_layers=2,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             type='speaker_token',
                             *args, **kwargs)