import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmallResNet2Blocks(nn.Module):
    def __init__(self, num_classes=10, resize_dim=64):
        super(SmallResNet2Blocks, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResNetBlock(64, 64, stride=1)
        self.layer2 = ResNetBlock(64, 128, stride=2)
        self.layer3 = ResNetBlock(128, 256, stride=2)
        self.fc = nn.Linear(256, num_classes)
        if resize_dim == 224:
            self.pool_dim = 56
        elif resize_dim == 128:
            self.pool_dim = 32
        elif resize_dim == 64:
            self.pool_dim = 16
        elif resize_dim == 32:
            self.pool_dim = 8
        else:
            raise ValueError("Invalid resize_dim")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, self.pool_dim)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, emb_dim, attn_drop, resid_drop, block_size):
        super().__init__()
        assert emb_dim % n_head == 0, 'emb_dim must be divisible by n_head'

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim)
        self.c_proj = nn.Linear(emb_dim, emb_dim)

        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))
        self.n_head = n_head
        self.emb_dim = emb_dim

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # split keys, querys and values in the last dimension
        q, k, v = self.c_attn(x).split(self.emb_dim, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, C // n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, C // n_head)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        att_before = F.softmax(att, dim=-1)
        att = self.attn_drop(att_before)
        y = att @ v # (B, n_head, T, T) x (B, n_head, T, C // n_head) -> (B, n_head, T, C // n_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        y = self.resid_drop(y)
        return y, att_before


class Block(nn.Module):
    def __init__(self, emb_dim, n_head, attn_drop, resid_drop, mlp_drop, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(n_head, emb_dim, attn_drop, resid_drop, block_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            NewGELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(mlp_drop)
        )

    def forward(self, x):
        x_att, att = self.attn(self.ln1(x))
        x = x + x_att
        x = x + self.mlp(self.ln2(x))
        return x, att


class GPT2(nn.Module):
    def __init__(self, 
                 block_size, 
                 vocab_size, 
                 emb_dim, 
                 n_head, 
                 n_layer, 
                 attn_drop, 
                 resid_drop, 
                 mlp_drop, 
                 resnet_embed="SmallResNet2Blocks",
                 resize_dim=64,
                 initialization="default",
                 std: float = 0.02,
                 a: float = -0.04,
                 b: float = 0.04):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
      
        if resnet_embed == "SmallResNet2Blocks":
            self.img_emb = nn.Linear(256, emb_dim)
            self.resnet_emb = SmallResNet2Blocks(resize_dim=resize_dim).cuda()
            self.resnet_emb.fc = torch.nn.Identity().cuda()
        elif resnet_embed == "ResNet18Pretrained":
            self.img_emb = nn.Linear(512, emb_dim)
            self.resnet_emb = models.resnet18(weights='ResNet18_Weights.DEFAULT').cuda()
            self.resnet_emb.fc = torch.nn.Identity().cuda()
        else:
            raise ValueError(f"Resnet Embedder {resnet_embed} is not supported.")
        
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[Block(emb_dim, n_head, attn_drop, resid_drop, mlp_drop, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self._initialization = initialization
        self._std = std
        self._a = a
        self._b = b

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                if self._initialization == "default":
                    # print("INFO: Using default initialization for c_proj.weight")
                    torch.nn.init.normal_(p, mean=0.0, std=self._std/math.sqrt(2*n_layer))
                else:
                    # print("INFO: Using trunc_normal_ initialization for c_proj.weight")
                    torch.nn.init.trunc_normal_(p, mean=0.0, std=self._std, a=self._a, b=self._b)


        self.attn_logits_tape = {}
        self.attn_weights_tape = {}
        self.attn_logits_grads = {}
            

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if self._initialization == "default":
                # print("INFO: Using default initialization for Linear and Embedding layers")
                module.weight.data.normal_(mean=0.0, std=self._std)
            else:
                # print("INFO: Using trunc_normal_ initialization for Linear and Embedding layers")
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=self._std, a=self._a, b=self._b)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

  
    def forward(self, batch):
        atts = []
    
        inputs, target = batch
        target = target.cuda()
        inputs = [self.img_emb(self.resnet_emb(inputs[i].cuda())) if i % 2 == 0 else self.tok_emb(inputs[i].cuda()) for i in range(len(inputs))]
        token = torch.stack(inputs, dim=1)

        
        pos = torch.arange(0, self.block_size, dtype=torch.long, device='cuda').unsqueeze(0) # shape (1, t)
        pos = self.pos_emb(pos)
        x = self.drop(token + pos)
        
        for blk in self.blocks:
            x, att = blk(x)
            atts.append(att)

        x = self.ln_f(x)
        logits = self.head(x)

        logits = logits.transpose(1, 2) # shape (B, num_classes, T) 

        loss = F.cross_entropy(logits[:,:, -1], target[:,-1], ignore_index=-1) 

        return logits[:,:, -1], target[:,-1], loss, atts
             