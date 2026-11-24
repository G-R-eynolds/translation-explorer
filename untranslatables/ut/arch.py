"""Where things live inside each architecture, so the rest of the code never touches model internals directly.

  text_model(model)   the decoder stack: model.model, or model.model.language_model for Gemma 3's multimodal
                      wrapper (Gemma3ForConditionalGeneration, which AutoModelForCausalLM returns for gemma-3-*-it)
  layers(model)       its decoder blocks;  final_norm(model)  its final norm
  tcfg(model)         the text config (Gemma 3 keeps num_hidden_layers / hidden_size in config.text_config)
  logits(model, h)    lm_head plus every post-processing step the model applies: Cohere's logit_scale, Gemma 2's
                      final_logit_softcapping
"""
import torch


def tcfg(model):
    return getattr(model.config, "text_config", None) or model.config


def text_model(model):
    m = model.model
    return getattr(m, "language_model", m)


def layers(model):
    return text_model(model).layers


def final_norm(model):
    return text_model(model).norm


def n_layers(model):
    return tcfg(model).num_hidden_layers


def hidden_size(model):
    return tcfg(model).hidden_size


def logits(model, h):
    lg = model.lm_head(h.to(model.lm_head.weight.dtype)).float()
    c = tcfg(model)
    if getattr(c, "model_type", "") == "cohere" or model.config.model_type == "cohere":
        lg = lg * getattr(c, "logit_scale", getattr(model.config, "logit_scale", 1.0))
    cap = getattr(c, "final_logit_softcapping", None)
    if cap:
        lg = torch.tanh(lg / cap) * cap
    return lg


def forward_hidden(model, **kw):
    """Run the decoder stack only (no lm_head); returns the HF output with .hidden_states / .last_hidden_state."""
    return text_model(model)(**kw)
