"""Steering from the cursor onward (spec §Method 5).

CursorHook adds a vector to the output of decoder block `layer` at every absolute position >= b, including positions
produced later by KV-cache decode steps (one token per call), or at position b only (only_at_b).  Absolute positions
are tracked with a counter: a call with sequence length > 1 is a prefill and resets it to that length; a length-1 call
is a decode step at position `pos`.  Rows of a batch must share the prompt (no left padding).
"""
import torch

from ut import arch


class CursorHook:
    def __init__(self, model, layer):
        self.vec, self.b, self.only, self.pos = None, 0, False, 0
        self.end_b = None
        self._explicit_pos = None
        self.h = arch.layers(model)[layer].register_forward_hook(self._hook)

    def set(self, vec=None, b=0, only_at_b=False, end_b=None):
        self.vec = None if vec is None else torch.as_tensor(vec)
        self.b, self.only, self.pos = int(b), bool(only_at_b), 0
        self.end_b = None if end_b is None else int(end_b)
        self._explicit_pos = None

    def sync_pos(self, pos: int):
        """Explicitly set the absolute start position of the next forward pass (for cropped/chunked KV caches)."""
        self.pos = int(pos)
        self._explicit_pos = int(pos)

    def remove(self):
        self.h.remove()

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        T = h.shape[1]
        if self._explicit_pos is not None:
            start = self._explicit_pos
            self._explicit_pos = None
            self.pos = start + T
        elif T > 1:
            start = 0
            self.pos = T
        else:
            start = self.pos
            self.pos += 1
        if self.vec is None:
            return None
        idx = torch.arange(start, start + T, device=h.device)
        if self.only:
            mask = idx == self.b
        elif self.end_b is not None:
            mask = (idx >= self.b) & (idx <= self.end_b)
        else:
            mask = idx >= self.b
        if not bool(mask.any()):
            return None
        add = mask[None, :, None].to(h.dtype) * self.vec.to(h.device, h.dtype)[None, None, :]
        new = h + add
        return (new, *out[1:]) if isinstance(out, tuple) else new
