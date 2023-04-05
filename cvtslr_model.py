import utils
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules import BiLSTMLayer, TemporalConv
from gloss_encoder import SelfAttentionAdapter


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class CVTSLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, args=None
             
    ):
        super(CVTSLRModel, self).__init__()
        self.loss = dict()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_backward_hook(self.backward_hook)

        self.GlossEncoder = SelfAttentionAdapter(hidden_size=hidden_size, num_layers=2, num_heads=4)

        self.gloss_linear1 = nn.Linear(hidden_size, self.num_classes)
        self.gloss_linear2 = nn.Linear(self.num_classes, hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.use_seqAE = args.use_seqAE
        if self.use_seqAE.lower()=="vae":
            self.bink_mean_linear = nn.Linear(hidden_size,hidden_size)
            self.bink_mean_var = nn.Linear(hidden_size,hidden_size)


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    # for AE during training only
    def repara(self, z_mean, z_log_var, sample=False):
        if sample:
            epsilon = torch.randn_like(z_mean)
            return z_mean + torch.exp(z_log_var / 2) * epsilon
        else:
            return z_mean + torch.exp(z_log_var / 2)

    def eval_network(self, x, len_x):
        if len(x.shape) == 5:
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        visual_features = x

        # Video-Gloss Adapter
        x_tmp = self.gloss_linear1(x)
        x = self.gloss_linear2(x_tmp)

        # Create masks for the self-attention Gloss encoder
        max_len = lgt.max().int()
        bt_size = lgt.numel()
        mask = torch.arange(0, max_len)
        mask_0 = mask.unsqueeze(0).expand(bt_size, max_len).lt(lgt.unsqueeze(1))
        mask = mask_0.unsqueeze(1) # B,1,L
        mask = mask.to(device=x.device)
        
        # Gloss Encoder
        x = x.transpose(0,1) #  T,B,D-> B,T,D 
        x = self.GlossEncoder(x, mask=mask.bool()) # B,T, D

        # set sample=False if not in VAE training stage
        if self.use_seqAE.lower()=="vae":
            z_mean = self.bink_mean_linear(x)
            z_log_var = self.bink_mean_var(x)
            x = self.repara(z_mean, z_log_var, sample=False)
            
        x = x.transpose(0,1) #  B,T,D -> T,B,D

        # Seq Decoder
        tm_outputs = self.temporal_model(x, lgt, enforce_sorted=True)
        outputs = self.classifier(tm_outputs['predictions']) # T,B,D -> T,B,C

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": visual_features,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "recognized_sents": pred
        }
