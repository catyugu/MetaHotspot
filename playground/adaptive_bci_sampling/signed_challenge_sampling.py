"""Full-support, signed-adversary guided task proposals for source-local MOR.

Discovery-only heuristic, never an acceptance certificate. Acceptance must use
the pre-existing independent full-time joint-risk challenge after freezing V.
"""
import numpy as np
import scipy.linalg as la


class SignedChallengeSampler:
    def __init__(self, ranges, G, c, rng, exploration=0.25, width=0.15,
                 history_size=16):
        self.ranges = np.asarray(ranges, float)
        self.rng = rng
        self.exploration = float(exploration)
        self.width = float(width)
        self.history_size = int(history_size)
        if not (0 < exploration <= 1 and width > 0 and history_size > 0):
            raise ValueError("invalid proposal mixture")
        self.lo = np.log(self.ranges[:, 0])
        self.span = np.log(self.ranges[:, 1] / self.ranges[:, 0])
        if np.any(self.span <= 0):
            raise ValueError("parameters require nonempty positive ranges")
        self.Q = G.T @ (G / np.asarray(c)[:, None])
        self.Q = (self.Q + self.Q.T) / 2
        la.cholesky(self.Q, lower=True)
        self.anchors = []
        self.counters = dict(explore=0, exploit=0, feedback=0,
                             residual_queries=0)

    def observe(self, h, residual_columns, D, inverse_lower=None, at_time=None):
        """A rejected challenge supplies source-local residual columns.

        The generalized eigenvector describes the signed *residual* direction
        for discovery. It is not an all-time error witness or acceptance test.
        """
        R = np.column_stack(residual_columns)
        WR = (R / np.sqrt(D)[:, None] if inverse_lower is None
              else inverse_lower.whiten(R))
        B = (WR.T @ WR)
        B = (B + B.T) / 2
        _, U = la.eigh(B, self.Q, check_finite=False)
        u = U[:, -1]
        sizes = np.linalg.norm(WR, axis=0)
        importance = (u * sizes) ** 2
        if not np.all(np.isfinite(importance)):
            raise ValueError("nonfinite task weights")
        p = len(importance)
        total = float(sum(importance))
        priority = (importance / total if total > 0
                    else np.ones(p) / p)
        priority = 0.9 * priority + 0.1 / p
        self.anchors.append((np.asarray(h, float).copy(), priority, at_time))
        self.anchors = self.anchors[-self.history_size:]
        self.counters["feedback"] += 1
        return priority

    def propose(self, local, plans):
        p = len(local)
        if not self.anchors or self.rng.random() < self.exploration:
            h = np.exp(self.lo + self.span *
                       self.rng.random(len(self.lo)))
            j = int(self.rng.integers(p))
            s = float(self.rng.choice(plans[j]))
            self.counters["explore"] += 1
        else:
            h0, weights, preferred_time = self.anchors[
                -1 - int(self.rng.integers(min(len(self.anchors), 8)))]
            relative = (np.log(h0) - self.lo) / self.span
            h = np.exp(self.lo + self.span *
                       np.clip(relative + self.width *
                               self.rng.normal(size=len(relative)), 0, 1))
            j = int(self.rng.choice(p, p=weights))
            values = local[j].residual(
                np.repeat(h[None, :], len(plans[j]), axis=0), plans[j])
            self.counters["residual_queries"] += len(plans[j])
            if preferred_time and self.rng.random() < 0.5:
                s = float(1. / preferred_time)
            else:
                s = float(plans[j][int(np.argmax(values))])
            self.counters["exploit"] += 1
        return h, j, s

    def choose_feedback(self, candidates):
        """Use every signed-source contribution, with positive exploration floor."""
        if not self.anchors:
            return max(candidates, key=lambda item: item[0])
        j = int(self.rng.choice(len(candidates), p=self.anchors[-1][1]))
        return candidates[j]
