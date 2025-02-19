import os.path as osp
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.layers.blocks import FrozenBatchNorm2d



from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    
    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model, ctx=8):
        super().__init__()
        n_cls = len(classnames)    # number of classes
        n_ctx_di = ctx             # number of context words in domain invariant part
        self.n_ctx_di = n_ctx_di
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        domain_names = ['clear', 'foggy']
        domain_templates = ['in an image' for domain_name in domain_names]
        n_dms = len(domain_names)  # number of domains
        n_ctx_ds = ctx             # number of context words in domain specific part
        self.n_dms = n_dms
        self.n_ctx_ds = n_ctx_ds
        n = n_ctx_di + n_ctx_ds    # number of context words in total

        prompt_prefix = ' '.join(['X'] * n)
        print(f'Initial context: "{prompt_prefix}"')

        print('Initializing a domain-invariant context')
        di_vectors = torch.empty(n_ctx_di, ctx_dim, dtype=dtype).to(torch.device("cuda"))

        nn.init.normal_(di_vectors, std=0.02)
        print(f'Number of domain-invariant context words (tokens): {n_ctx_di}')       
        self.ctx_di = nn.Parameter(di_vectors)
        # EMA
        self.register_buffer('ctx_di_ema', di_vectors.clone().detach())


        print('Initializing a domain-specific context')
        ds_vectors = torch.empty(n_dms, n_ctx_ds, ctx_dim, dtype=dtype).to(torch.device("cuda"))

        nn.init.normal_(ds_vectors, std=0.02)
        print(f'Number of domain-specific context words (tokens): {n_ctx_ds}')
        self.ctx_ds = nn.Parameter(ds_vectors)
        # EMA
        self.register_buffer('ctx_ds_ema', ds_vectors.clone().detach())

        classnames = [name.replace('_', ' ') for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + ' ' + name + ' ' + domain + '.' for domain in domain_templates for name in classnames]
        print(f'Prompts: {prompts}')

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(torch.device("cuda"))
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer('token_prefix', embedding[:, :1, :]) # SOS
        self.register_buffer('token_suffix', embedding[:, 1 + n:, :]) # CLS, DMS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts # torch.Tensor
        self.name_lens = name_lens

        self.clip_model = clip_model
    
    # lora prompt tuning version
    def forward(self, x_di, x_ds):
        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx_di = x_di.half().reshape(self.n_ctx_di, -1)        # [8, 512]
        ctx_dim = ctx_di.size(-1)
        ctx_ds = x_ds.half().reshape(self.n_ctx_ds, -1)        # [8, 512]

        if ctx_di.dim() == 2:
            ctx_di = ctx_di.unsqueeze(0).expand(self.n_dms, -1, -1) # [n_dms, 8, 512]
            ctx_di = ctx_di.unsqueeze(1).expand(-1, self.n_cls, -1, -1) # [n_dms, n_cls, 8, 512]
        if ctx_ds.dim() == 2:
            ctx_ds = ctx_ds.unsqueeze(0).expand(self.n_dms, -1, -1) # [n_dms, 8, 512]
            ctx_ds = ctx_ds.unsqueeze(1).expand(-1, self.n_cls, -1, -1) # [n_dms, n_cls, 8, 512]

        ctx = torch.cat([ctx_di, ctx_ds], dim=2).reshape(self.n_dms * self.n_cls, self.n_ctx_di + self.n_ctx_ds, ctx_dim) # [n_dms, n_cls, 16, 512]-> [n_dms * n_cls, 16, 512]
        prompts = torch.cat([
            prefix, # [n_dms * n_cls, 1, 512]
            ctx,    # [n_dms * n_cls, 16, 512]
            suffix  # [n_dms * n_cls, *, 512]
        ], dim=1)

        # EMA
        prompts_ema = []
        
        return prompts, self.tokenized_prompts, prompts_ema

# lora prompt tuning version
class DAPromptHead(nn.Module):

    def __init__(self, classnames, clip_model, ctx=8):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model,ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        
        self.dtype = clip_model.dtype

    
    def get_embedding(self, x_di, x_ds):
        prompts, tokenized_prompts, _ = self.prompt_learner(x_di, x_ds)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        return text_features.float()

    def get_embedding_ema(self, x_di, x_ds):
        _, tokenized_prompts, prompts = self.prompt_learner(x_di, x_ds)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        return text_features.float()

    def forward(self, image_features, x_di, x_ds):
        text_features = self.get_embedding(x_di, x_ds)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.t()
        return logits
