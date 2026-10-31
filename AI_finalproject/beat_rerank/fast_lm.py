"""KV-cached decoding for UniMuMo's music-motion LM.

UniMuMo's LMModel.generate re-runs the whole transformer on the full prefix at every step
(no cache), so a 10 s clip (503 steps) costs O(n^2). Its self-attention mask is causal in
time across both streams (music token t and motion token t both see every token of both
streams up to time t), so at each step only the two new tokens need to be computed, and
they may attend to every cached token. This module re-implements one decoding step with a
key/value cache, reusing the model's own embeddings, fuser, norms, cross-attention,
modality-specific feed-forward blocks and output heads. verify_cached_step() checks the
cached logits against the original full forward pass.
"""
import math
import types

import torch
import torch.nn.functional as F
from einops import rearrange


def _attn_params(attn):
    if attn.custom:
        return attn.in_proj_weight, attn.in_proj_bias, attn.out_proj
    return attn.mha.in_proj_weight, attn.mha.in_proj_bias, attn.mha.out_proj


class KVCache:
    def __init__(self, lm, batch, max_len, device, dtype):
        attn = lm.transformer.layers[0].self_attn
        h = attn.num_heads
        d = attn.embed_dim // h
        n_layers = len(lm.transformer.layers)
        self.k = torch.zeros(n_layers, batch, h, max_len, d, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.len = 0


def _self_attn_step(attn, h, cache, li):
    w, b, out_proj = _attn_params(attn)
    assert not attn.rope and not attn.qk_layer_norm and attn.kv_repeat == 1
    q, k, v = rearrange(F.linear(h, w, b), "b t (p h d) -> p b h t d", p=3, h=attn.num_heads)
    n0, n = cache.len, h.shape[1]
    cache.k[li, :, :, n0:n0 + n] = k
    cache.v[li, :, :, n0:n0 + n] = v
    K = cache.k[li, :, :, :n0 + n]
    V = cache.v[li, :, :, :n0 + n]
    if attn.attention_as_float32:
        q, K, V = q.float(), K.float(), V.float()
    att = torch.softmax((q @ K.transpose(-2, -1)) / math.sqrt(q.shape[-1]), dim=-1)
    x = rearrange((att @ V).to(h.dtype), "b h t d -> b t (h d)")
    return out_proj(x)


def _layer_step(layer, x, cache, li, cross_src, cross_mask):
    assert layer.norm_first
    x = x + layer.layer_scale_1(layer.dropout1(_self_attn_step(layer.self_attn, layer.norm1(x), cache, li)))
    if cross_src is not None:
        x = x + layer.layer_scale_cross(layer._cross_attention_block(layer.norm_cross(x), cross_src, cross_mask))
    x_mu, x_mo = x[:, :1], x[:, 1:]
    x_mu = x_mu + layer.layer_scale_2(layer._ff_block(layer.norm2(x_mu)))
    x_mo = x_mo + layer.layer_scale_2(layer._ff_block_motion(layer.norm2_motion(x_mo)))
    return torch.cat((x_mu, x_mo), dim=1)


def cached_step(lm, music_tok, motion_tok, pos, cache, condition_tensors, cross_mask):
    """music_tok, motion_tok: (B, K, 1) tokens at time `pos`; returns logits (B, K, card) x 2."""
    from unimumo.audio.audiocraft_.modules.transformer import create_sin_embedding
    K = lm.num_codebooks
    x_mu = sum(lm.emb[k](music_tok[:, k]) for k in range(K))
    x_mo = sum(lm.motion_emb[k](motion_tok[:, k]) for k in range(K))
    x = torch.cat((x_mu, x_mo), dim=1)  # (B, 2, D)
    x, cross_src = lm.fuser(x, condition_tensors)
    tr = lm.transformer
    assert tr.positional_embedding == "sin"
    positions = torch.full((1, 1, 1), pos, device=x.device, dtype=torch.long)
    pe = tr.positional_scale * create_sin_embedding(positions, x.shape[-1], max_period=tr.max_period, dtype=x.dtype)
    x = x + pe
    for li, layer in enumerate(tr.layers):
        x = _layer_step(layer, x, cache, li, cross_src, cross_mask)
    cache.len += 2
    if lm.out_norm:
        x = lm.out_norm(x)
    mu_logits = torch.stack([lm.linears[k](x[:, 0]) for k in range(K)], dim=1)
    mo_logits = torch.stack([lm.motion_linears[k](x[:, 1]) for k in range(K)], dim=1)
    return mu_logits, mo_logits


def _sample(logits, use_sampling, temp, top_k, top_p):
    from unimumo.audio.audiocraft_.utils import utils
    if use_sampling and temp > 0.0:
        probs = torch.softmax(logits / temp, dim=-1)
        if top_p > 0.0:
            return utils.sample_top_p(probs, p=top_p)
        if top_k > 0:
            return utils.sample_top_k(probs, k=top_k)
        return utils.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _build_cfg_conditions(lm, conditions, device):
    """Same as the beginning of UniMuMo's LMModel.generate."""
    from unimumo.audio.audiocraft_.modules.conditioners import ClassifierFreeGuidanceDropout
    num_samples = len(conditions) // 2
    music_conditions = conditions[:num_samples]
    motion_conditions = conditions[num_samples:]
    null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
    music_conditions = music_conditions + null_conditions[:num_samples]
    motion_conditions = motion_conditions + null_conditions[num_samples:]
    ct_music = lm.condition_provider(lm.condition_provider.tokenize(music_conditions, device))
    ct_motion = lm.condition_provider(lm.condition_provider.tokenize(motion_conditions, device))
    cond = torch.cat([ct_music["description"][0], ct_motion["description"][0]], dim=1)
    mask = torch.zeros((cond.shape[0], 2, cond.shape[-2]), dtype=torch.bool, device=device)
    m_mu = ct_music["description"][1]
    m_mo = ct_motion["description"][1]
    mask[:, 0, :m_mu.shape[-1]] = m_mu.bool()
    mask[:, 1, m_mu.shape[-1]:] = m_mo.bool()
    return {"description": (cond, mask)}


@torch.no_grad()
def generate_cached(self, conditions=[], mode=None, music_code=None, motion_code=None, num_samples=None,
                    max_gen_len=256, use_sampling=True, temp=1.0, top_k=250, top_p=0.0, cfg_coef=None,
                    check=False):
    """Drop-in replacement for UniMuMo's LMModel.generate (same sampling, with a KV cache)."""
    assert not self.training
    device = next(iter(self.parameters())).device
    assert mode in ["music_motion", "music2motion", "motion2music"]
    assert music_code is None or motion_code is None
    cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
    B, K = len(conditions) // 2, self.num_codebooks
    cfg_conditions = _build_cfg_conditions(self, conditions, device)
    cross_mask = torch.where(cfg_conditions["description"][-1], 0.0, float("-inf"))

    pattern = self.pattern_provider.get_pattern(max_gen_len)
    unknown = -1
    mu_codes = music_code if music_code is not None else torch.full((B, K, max_gen_len), unknown, dtype=torch.long, device=device)
    mo_codes = motion_code if motion_code is not None else torch.full((B, K, max_gen_len), unknown, dtype=torch.long, device=device)
    mu_seq, _, mu_mask = pattern.build_pattern_sequence(mu_codes, self.special_token_id)
    mo_seq, _, mo_mask = pattern.build_pattern_sequence(mo_codes, self.special_token_id)
    S = mu_seq.shape[-1]
    dtype = next(iter(self.parameters())).dtype
    cache = KVCache(self, 2 * B, 2 * S, device, dtype)

    for offset in range(1, S):
        tok_mu = mu_seq[..., offset - 1:offset]
        tok_mo = mo_seq[..., offset - 1:offset]
        mu_logits, mo_logits = cached_step(self, torch.cat([tok_mu, tok_mu]), torch.cat([tok_mo, tok_mo]),
                                           offset - 1, cache, cfg_conditions, cross_mask)
        mu_c, mu_u = mu_logits.split(B, dim=0)
        mo_c, mo_u = mo_logits.split(B, dim=0)
        nxt_mu = _sample(mu_u + (mu_c - mu_u) * cfg_coef, use_sampling, temp, top_k, top_p)
        nxt_mo = _sample(mo_u + (mo_c - mo_u) * cfg_coef, use_sampling, temp, top_k, top_p)
        nxt_mu[~mu_mask[..., offset:offset + 1].expand(B, -1, -1)] = self.special_token_id
        nxt_mo[~mo_mask[..., offset:offset + 1].expand(B, -1, -1)] = self.special_token_id
        if music_code is None:
            cur = mu_seq[..., offset:offset + 1]
            mu_seq[..., offset:offset + 1] = torch.where(cur == unknown, nxt_mu, cur)
        if motion_code is None:
            cur = mo_seq[..., offset:offset + 1]
            mo_seq[..., offset:offset + 1] = torch.where(cur == unknown, nxt_mo, cur)

    assert not (mu_seq == unknown).any() and not (mo_seq == unknown).any()
    mu_out, _, _ = pattern.revert_pattern_sequence(mu_seq, special_token=unknown)
    mo_out, _, _ = pattern.revert_pattern_sequence(mo_seq, special_token=unknown)
    assert (mu_out[..., :max_gen_len] != unknown).all() and (mo_out[..., :max_gen_len] != unknown).all()
    return mu_out[..., :max_gen_len], mo_out[..., :max_gen_len]


def enable_kv_cache(unimumo_model):
    lm = unimumo_model.music_motion_lm.model
    lm.generate = types.MethodType(generate_cached, lm)
    return unimumo_model


@torch.no_grad()
def verify_cached_step(unimumo_model, text, n_steps=40, mode="music_motion", seed=0):
    """Max abs difference between cached logits and the original full forward, over n_steps."""
    from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes, WavCondition
    mm = unimumo_model.music_motion_lm
    lm = mm.model
    device = next(iter(lm.parameters())).device
    attrs = mm._prepare_tokens_and_attributes([text], mode=mode)
    cfg_conditions = _build_cfg_conditions(lm, attrs, device)
    cross_mask = torch.where(cfg_conditions["description"][-1], 0.0, float("-inf"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = lm.num_codebooks
    mu = torch.randint(0, lm.card, (1, K, n_steps), generator=g).to(device)
    mo = torch.randint(0, lm.card, (1, K, n_steps), generator=g).to(device)
    mu[..., 0] = lm.special_token_id
    mo[..., 0] = lm.special_token_id
    cache = KVCache(lm, 2, 2 * n_steps, device, next(iter(lm.parameters())).dtype)
    worst = 0.0
    for t in range(n_steps):
        c_mu, c_mo = cached_step(lm, torch.cat([mu[..., t:t + 1]] * 2), torch.cat([mo[..., t:t + 1]] * 2),
                                 t, cache, cfg_conditions, cross_mask)
        if t in (0, n_steps // 2, n_steps - 1):
            seq = torch.cat((mu[..., :t + 1], mo[..., :t + 1]), dim=-1)
            seq = torch.cat([seq, seq], dim=0)
            src_mask = lm.get_self_attn_mask(t + 1, t + 1, mode=mode)
            f_mu, f_mo = lm(seq, conditions=[], condition_tensors=cfg_conditions, src_mask=src_mask,
                            cross_attn_mask=cross_mask)
            d = max((f_mu[:, :, -1] - c_mu).abs().max().item(), (f_mo[:, :, -1] - c_mo).abs().max().item())
            scale = f_mu[:, :, -1].abs().max().item()
            worst = max(worst, d)
            print(f"step {t}: max |full - cached| = {d:.2e} (logit scale {scale:.1f})")
    return worst
