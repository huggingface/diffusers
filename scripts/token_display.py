"""Live terminal token-grid display for discrete diffusion pipelines.

Works with any diffusers pipeline that supports `callback_on_step_end`.
Pass an instance as the callback -- it auto-detects DFlash vs LLaDA2 from
which kwargs are present.

Color scheme (colorblind-safe, no red/green):
  dim grey  ░   = mask token (not yet committed)
  dim blue      = draft proposal (DFlash: block proposed, not yet verified)
  bold yellow   = newly committed / just accepted this step
  dim cyan      = inside <think>...</think> block
  white         = previously committed

DFlash two-phase flash per block:
  1. Full draft block appears dim blue (the proposal, all positions at once)
  2. 150ms pause
  3. Accepted tokens snap to bold yellow, rejected positions return to ░

Falls back to plain print if `rich` is not installed or stdout is not a TTY.
"""

from __future__ import annotations

import time
from collections import Counter


def _try_import_rich():
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.text import Text

        return Console, Live, Panel, Text
    except ImportError:
        return None


class TokenDisplay:
    """Shared live display for DFlash, LLaDA2, and future discrete diffusion pipelines."""

    def __init__(
        self,
        tokenizer,
        mask_token_id: int | None,
        title: str = "Diffusion",
        draft_pause: float = 0.15,
        block_size: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.title = title
        self.draft_pause = draft_pause  # seconds to hold draft frame before snapping to verified
        self.block_size = block_size  # if set, only show committed + one block ahead
        self._rich = _try_import_rich()
        self._live = None
        self._is_tty: bool = False
        self._t0: float = 0.0
        self._step: int = 0
        self._prev_committed: set[int] = set()
        self._num_prompt_tokens: int | None = None
        self._in_think: bool = False
        self._decoded_so_far: str = ""

    # ------------------------------------------------------------------ context manager

    def start(self) -> "TokenDisplay":
        import sys

        self._t0 = time.time()
        self._step = 0
        self._prev_committed = set()
        self._num_prompt_tokens = None
        self._in_think = False
        self._decoded_so_far = ""
        self._is_tty = sys.stdout.isatty()
        if self._rich and self._is_tty:
            Console, Live, Panel, Text = self._rich
            self._live = Live(
                Console(highlight=False),
                refresh_per_second=15,
                transient=False,
                auto_refresh=False,
            )
            self._live.start()
        return self

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __enter__(self) -> "TokenDisplay":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------ mask detection

    def _discover_mask_token(self, tokens: list[int]) -> None:
        """If mask_token_id wasn't provided or doesn't appear, infer it from the buffer tail."""
        if self.mask_token_id is not None:
            pos = next((i for i, t in enumerate(tokens) if t == self.mask_token_id), None)
            if pos is not None:
                self._num_prompt_tokens = pos
                return
        # fallback: the most frequent token in the second half is almost certainly the mask
        half = max(len(tokens) // 2, 1)
        most_common = Counter(tokens[half:]).most_common(1)
        if most_common:
            self.mask_token_id = most_common[0][0]
            pos = next((i for i, t in enumerate(tokens) if t == self.mask_token_id), len(tokens))
            self._num_prompt_tokens = pos
        else:
            self._num_prompt_tokens = len(tokens)

    # ------------------------------------------------------------------ rendering

    def _style_for(self, tok_id: int, is_new: bool) -> str:
        word = self.tokenizer.decode([tok_id], skip_special_tokens=False)
        self._decoded_so_far += word
        think_opens = self._decoded_so_far.count("<think>")
        think_closes = self._decoded_so_far.count("</think>")
        self._in_think = think_opens > think_closes
        if is_new:
            return "bold yellow"
        return "dim cyan" if self._in_think else "white"

    def _render(
        self,
        gen_tokens: list[int],
        new_indices: set[int],
        draft_indices: set[int] | None = None,
        subtitle_extra: str = "",
    ) -> None:
        elapsed = time.time() - self._t0
        ops = self._step / elapsed if elapsed > 0 else 0.0
        committed = sum(1 for t in gen_tokens if t != self.mask_token_id)
        stats = (
            f"step {self._step} | {elapsed:.1f}s | {ops:.1f} OPS"
            f" | +{len(new_indices)} | {committed} committed"
            + (f" | {subtitle_extra}" if subtitle_extra else "")
        )

        if self._rich and self._is_tty and self._live is not None:
            _, _, Panel, Text = self._rich
            text = Text()
            self._decoded_so_far = ""
            self._in_think = False
            for i, tok_id in enumerate(gen_tokens):
                if tok_id == self.mask_token_id:
                    text.append("░ ", style="dim")
                else:
                    is_draft = draft_indices is not None and i in draft_indices
                    style = "dim blue" if is_draft else self._style_for(tok_id, i in new_indices)
                    word = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                    text.append(word + " ", style=style)
            self._live.update(
                Panel(text, title=f"[cyan bold]{self.title}[/]", subtitle=f"[dim]{stats}[/]")
            )
            self._live.refresh()
        else:
            if new_indices:
                new_words = "".join(
                    self.tokenizer.decode([gen_tokens[i]], skip_special_tokens=False)
                    for i in sorted(new_indices)
                    if gen_tokens[i] != self.mask_token_id
                )
                print(f"[{self._step:3d} | {elapsed:.1f}s] {new_words}", flush=True)

    # ------------------------------------------------------------------ pipeline callbacks

    def __call__(self, pipe, step: int, timestep, kwargs: dict) -> dict:
        self._step = step
        if "output_ids" in kwargs:
            return self._dflash_step(kwargs)
        if "block_x" in kwargs:
            return self._llada2_step(kwargs)
        return {}

    def _dflash_step(self, kwargs: dict) -> dict:
        tokens = kwargs["output_ids"][0].tolist()
        if self._num_prompt_tokens is None:
            self._discover_mask_token(tokens)

        gen_full = tokens[self._num_prompt_tokens :]
        cur = {i for i, t in enumerate(gen_full) if t != self.mask_token_id}
        new = cur - self._prev_committed

        block_ids = kwargs.get("block_output_ids")
        accepted_length = kwargs.get("accepted_length")
        block_size = len(block_ids[0]) if block_ids is not None else 16

        # only show committed tokens + one block of ░ ahead (not the full pre-allocated buffer)
        visible_end = len(cur) + block_size
        gen = gen_full[:visible_end]

        if block_ids is not None and self._is_tty:
            # Phase 1: show the full draft proposal (dim blue) at the block position
            draft_start = len(self._prev_committed)
            draft_tokens = block_ids[0].tolist()

            gen_with_draft = list(gen)
            for j, tok in enumerate(draft_tokens):
                pos = draft_start + j
                if pos < len(gen_with_draft):
                    gen_with_draft[pos] = tok

            draft_indices = set(range(draft_start, draft_start + len(draft_tokens)))
            al = int(accepted_length[0].item()) if accepted_length is not None else len(draft_tokens)
            self._render(
                gen_with_draft,
                new_indices=set(),
                draft_indices=draft_indices,
                subtitle_extra=f"drafting {len(draft_tokens)} → accepting {al + 1}",
            )
            time.sleep(self.draft_pause)

        # Phase 2: snap to verified state (accepted = yellow, rejected snaps back to ░)
        self._prev_committed = cur
        self._render(gen, new)
        return {}

    def _llada2_step(self, kwargs: dict) -> dict:
        block_x = kwargs["block_x"]
        transfer_index = kwargs.get("transfer_index")

        tokens_full = block_x[0].tolist()
        if self._num_prompt_tokens is None:
            self._discover_mask_token(tokens_full)

        gen_full = tokens_full[self._num_prompt_tokens :]

        if transfer_index is not None:
            transferred = transfer_index[0, self._num_prompt_tokens :].tolist()
            new = {i for i, v in enumerate(transferred) if v}
            self._prev_committed |= new
        else:
            cur = {i for i, t in enumerate(gen_full) if t != self.mask_token_id}
            new = cur - self._prev_committed
            self._prev_committed = cur

        # only render committed + one active block (if block_size was given)
        if self.block_size:
            committed_end = max(self._prev_committed, default=-1) + 1 if self._prev_committed else 0
            visible_end = committed_end + self.block_size
            gen = gen_full[:visible_end]
        else:
            gen = gen_full

        self._render(gen, new)
        return {}
