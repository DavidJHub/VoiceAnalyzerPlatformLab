#!/usr/bin/env python
"""
CRF sequence decoder for label sequence optimisation.

Transition probabilities are estimated from observed label sequences in
training data (not end-to-end gradient descent).  At inference, emissions
come from the BERT + time-fusion log-probabilities produced by fittingDeep,
and Viterbi decoding finds the globally optimal label sequence for the call.

Dependency:  pip install torchcrf>=1.1.0
"""
import json
import numpy as np
import torch
from torchCRF import CRF


class CRFSequenceDecoder:
    """
    Wraps torchcrf.CRF with helpers for:
      - Fitting transition/start/end weights from training label sequences.
      - Viterbi decoding over externally-computed log-probability emissions.
      - Serialising/deserialising state to/from a plain dict (JSON-safe).
    """

    def __init__(self, num_tags: int, labels: list = None):
        """
        Args:
            num_tags : number of distinct labels (must match label order used
                       throughout training and fitting).
            labels   : optional list of label strings for debug prints.
        """
        self.num_tags = num_tags
        self.labels = labels or [str(i) for i in range(num_tags)]
        self.crf = CRF(num_tags, batch_first=False)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit_transitions(self, label_sequences: list, laplace: float = 1.0):
        """
        Estimate CRF transition matrix (and start/end biases) from observed
        label sequences using maximum-likelihood + Laplace smoothing.

        Args:
            label_sequences : list of lists of int label indices; one list
                              per call, ordered by turn/window index.
            laplace         : Laplace (add-k) smoothing constant.
        """
        trans  = torch.zeros(self.num_tags, self.num_tags) + laplace
        start_ = torch.zeros(self.num_tags) + laplace
        end_   = torch.zeros(self.num_tags) + laplace

        for seq in label_sequences:
            if not seq:
                continue
            start_[seq[0]] += 1
            end_[seq[-1]]  += 1
            for i in range(len(seq) - 1):
                trans[seq[i], seq[i + 1]] += 1

        log_trans  = torch.log(trans  / trans.sum(dim=1, keepdim=True))
        log_start  = torch.log(start_ / start_.sum())
        log_end    = torch.log(end_   / end_.sum())

        with torch.no_grad():
            self.crf.transitions.data.copy_(log_trans)
            self.crf.start_transitions.data.copy_(log_start)
            self.crf.end_transitions.data.copy_(log_end)

        self._fitted = True

        print(f"[CRF] Transitions fitted from {len(label_sequences)} call sequence(s).")
        print("[CRF] Top-3 most likely successors per label:")
        for i in range(self.num_tags):
            top_vals, top_idx = torch.topk(log_trans[i], min(3, self.num_tags))
            successors = ", ".join(
                f"{self.labels[j]}({v:.2f})"
                for j, v in zip(top_idx.tolist(), top_vals.tolist())
            )
            print(f"  {self.labels[i]:25s} -> {successors}")

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, log_probs_seq: np.ndarray) -> list:
        """
        Viterbi decoding over a pre-computed emission matrix.

        Args:
            log_probs_seq : np.ndarray shape [seq_len, num_tags] containing
                            log-probabilities (e.g. from BERT + time fusion).

        Returns:
            list of int label indices representing the optimal Viterbi path.
            Falls back to per-step argmax if CRF has not been fitted.
        """
        if not self._fitted:
            return list(np.argmax(log_probs_seq, axis=1))

        # torchcrf expects shape [seq_len, batch_size, num_tags]
        emissions = (
            torch.tensor(log_probs_seq, dtype=torch.float32)
            .unsqueeze(1)  # batch_size = 1
        )
        mask = torch.ones(emissions.shape[0], 1, dtype=torch.bool)

        with torch.no_grad():
            result = self.crf.decode(emissions, mask=mask)

        return result[0]  # unwrap single-batch result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the CRF state."""
        return {
            "num_tags": self.num_tags,
            "labels": self.labels,
            "transitions": self.crf.transitions.data.tolist(),
            "start_transitions": self.crf.start_transitions.data.tolist(),
            "end_transitions": self.crf.end_transitions.data.tolist(),
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CRFSequenceDecoder":
        """Reconstruct a CRFSequenceDecoder from a dict produced by to_dict()."""
        obj = cls(num_tags=d["num_tags"], labels=d.get("labels"))
        obj.crf = CRF(d["num_tags"], batch_first=False)
        with torch.no_grad():
            obj.crf.transitions.data.copy_(torch.tensor(d["transitions"]))
            obj.crf.start_transitions.data.copy_(torch.tensor(d["start_transitions"]))
            obj.crf.end_transitions.data.copy_(torch.tensor(d["end_transitions"]))
        obj._fitted = d.get("fitted", True)
        return obj

    def save(self, path: str):
        """Persist state to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CRFSequenceDecoder":
        """Load state from a JSON file created by save()."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
