#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""Evaluation and verification code for QSS.

- Unlearning correctness (bitwise identity vs retrain-from-scratch)
- Retention curves (accuracy vs deletion fraction)
- LiRA membership inference attack (Carlini et al. 2022)
- Codebook fragility (LOO code-flip rate)
"""

from __future__ import annotations

import numpy as np

from qss import (
    build_accumulators,
    encode_rq,
    l2_normalize,
    QSSModel,
    qss_model_predict,
    qss_train,
    qss_unlearn,
    rematerialize,
    train_rvq,
)


# ── Unlearning Verification ──


def verify_unlearn_correctness(
    embeddings: np.ndarray,
    labels: np.ndarray,
    K: int,
    b: int,
    ell: int,
    n_delete: int = 100,
    seed: int = 42,
) -> dict:
    """Verify that qss_unlearn() produces bitwise-identical accumulators
    to retraining from scratch on the remaining data.

    Returns dict with match status and max differences across all
    accumulators (S0, n, N_co).
    """
    emb = l2_normalize(embeddings)
    rq = train_rvq(emb, emb.shape[1], b, ell)
    all_codes = encode_rq(rq, emb, b, ell)
    model = qss_train(emb, labels, K, b, ell, rq=rq)

    rng = np.random.RandomState(seed)
    del_idx = rng.choice(len(labels), n_delete, replace=False)
    del_codes = all_codes[del_idx]
    del_labels = labels[del_idx]

    model_unl = qss_unlearn(model, del_codes, del_labels)

    remaining = np.ones(len(labels), dtype=bool)
    remaining[del_idx] = False
    S0_ret, n_ret, N_ret = build_accumulators(
        all_codes[remaining], labels[remaining], K, b, ell,
    )
    mu_ret = rematerialize(S0_ret, n_ret, N_ret, K, b, ell)

    all_match = True
    max_s0_diff = 0.0
    max_n_diff = 0
    max_pred_diff = 0.0
    for m in range(ell):
        s0_diff = float(np.abs(model_unl.S0[m] - S0_ret[m]).max())
        n_diff = int(np.abs(model_unl.n[m] - n_ret[m]).max())
        pred_diff = float(np.abs(model_unl.mu[m] - mu_ret[m]).max())
        max_s0_diff = max(max_s0_diff, s0_diff)
        max_n_diff = max(max_n_diff, n_diff)
        max_pred_diff = max(max_pred_diff, pred_diff)
        if s0_diff > 0 or n_diff > 0:
            all_match = False

    max_joint_diff = 0
    for m in range(ell):
        for k in range(m):
            j_diff = int(np.abs(model_unl.N_co[(k, m)] - N_ret[(k, m)]).max())
            max_joint_diff = max(max_joint_diff, j_diff)
            if j_diff > 0:
                all_match = False

    return {
        "bitwise_identical": all_match,
        "max_S0_diff": max_s0_diff,
        "max_n_diff": max_n_diff,
        "max_joint_diff": max_joint_diff,
        "max_pred_diff": max_pred_diff,
    }


# ── Retention Curves ──


def retention_curve(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    K: int,
    b: int,
    ell: int,
    fractions: list[float] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Accuracy at increasing deletion fractions, with accumulator identity check.

    At each fraction, verifies that the unlearned accumulators (S0, n, N_co)
    are exactly equal to those built from scratch on the remaining data.

    Returns list of dicts per fraction.
    """
    if fractions is None:
        fractions = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    train_emb = l2_normalize(train_emb)
    val_emb = l2_normalize(val_emb)

    rq = train_rvq(train_emb, train_emb.shape[1], b, ell)
    all_codes = encode_rq(rq, train_emb, b, ell)
    val_codes = encode_rq(rq, val_emb, b, ell)
    model = qss_train(train_emb, train_labels, K, b, ell, rq=rq)

    rng = np.random.RandomState(seed)
    results = []

    for frac in fractions:
        n_del = int(len(train_labels) * frac)
        if n_del == 0 and frac > 0:
            continue

        del_idx = (
            rng.choice(len(train_labels), n_del, replace=False)
            if n_del > 0
            else np.array([], dtype=np.int64)
        )

        if n_del > 0:
            model_unl = qss_unlearn(
                model, all_codes[del_idx], train_labels[del_idx],
            )
        else:
            model_unl = model
        scores_unl = qss_model_predict(model_unl, val_emb)
        acc_unl = float((scores_unl.argmax(axis=1) == val_labels).mean())

        remaining = np.ones(len(train_labels), dtype=bool)
        if n_del > 0:
            remaining[del_idx] = False
        S0r, nr, Nr = build_accumulators(
            all_codes[remaining], train_labels[remaining], K, b, ell,
        )
        mu_r = rematerialize(S0r, nr, Nr, K, b, ell)
        scores_ret = np.zeros((len(val_labels), K), dtype=np.float64)
        for lev in range(ell):
            scores_ret += mu_r[lev][val_codes[:, lev]]
        acc_ret = float((scores_ret.argmax(axis=1) == val_labels).mean())

        accum_match = True
        max_abs_diff = 0.0
        for m in range(ell):
            s0_diff = float(np.abs(model_unl.S0[m] - S0r[m]).max())
            n_diff = int(np.abs(model_unl.n[m] - nr[m]).max())
            max_abs_diff = max(max_abs_diff, s0_diff)
            if s0_diff > 0 or n_diff > 0:
                accum_match = False
        for m in range(ell):
            for k in range(m):
                j_diff = int(np.abs(model_unl.N_co[(k, m)] - Nr[(k, m)]).max())
                if j_diff > 0:
                    accum_match = False

        results.append({
            "frac": frac,
            "n_deleted": n_del,
            "acc_unlearn": acc_unl,
            "acc_retrain": acc_ret,
            "accumulators_identical": accum_match,
            "max_abs_diff": max_abs_diff,
        })

    return results


# ── LiRA (Membership Inference Attack) ──


def _qss_confidence(
    mu: list[np.ndarray],
    codes_row: np.ndarray,
    true_class: int,
    K: int,
    ell: int,
) -> float:
    """Logit of true class: log p(y|x) - log(1 - p(y|x)).

    Follows Carlini et al. 2022 — the rescaled logit is the standard
    LiRA signal.
    """
    score = np.zeros(K, dtype=np.float64)
    for level in range(ell):
        score += mu[level][codes_row[level]]
    score -= score.max()
    exp_score = np.exp(score)
    prob = exp_score / exp_score.sum()
    p = np.clip(prob[true_class], 1e-15, 1 - 1e-15)
    return float(np.log(p) - np.log(1 - p))


def lira_mia(
    embeddings: np.ndarray,
    labels: np.ndarray,
    K: int,
    b: int,
    ell: int,
    n_shadow: int = 64,
    n_audit: int = 1000,
    seed: int = 42,
) -> dict:
    """LiRA membership inference attack against QSS (Carlini et al. 2022).

    Protocol:
    1. Select n_audit audit points. Split the remaining data into a
       held-out "target" training set (50%) and a pool for shadows.
    2. Train the TARGET QSS model on the target set — audit points that
       fall inside the target set are "members", the rest are "non-members".
    3. Train n_shadow shadow QSS models, each on a fresh random 50% subset
       of all non-audit data. For each audit point and each shadow, compute
       the model confidence (rescaled logit of true class) with that point
       IN vs OUT of the shadow's training set.
    4. Per audit point, fit Gaussians to the IN-confidences and OUT-confidences
       across shadows. Compute the log-likelihood ratio of the target model's
       confidence under the IN vs OUT Gaussian.
    5. Evaluate: members should have higher LR scores than non-members.

    Returns dict with AUC and TPR at FPR thresholds {0.1%, 1%, 10%}.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    N = len(labels)
    emb = l2_normalize(embeddings)
    rq = train_rvq(emb, emb.shape[1], b, ell)
    all_codes = encode_rq(rq, emb, b, ell)

    audit_idx = rng.choice(N, min(n_audit, N), replace=False)
    audit_set = set(audit_idx.tolist())

    non_audit_idx = np.array([i for i in range(N) if i not in audit_set])
    target_size = len(non_audit_idx) // 2
    target_pool = rng.choice(non_audit_idx, target_size, replace=False)
    target_set = set(target_pool.tolist())

    is_member = np.array([idx in target_set for idx in audit_idx])

    target_codes = all_codes[target_pool]
    target_labels = labels[target_pool]
    S0_t, n_t, N_t = build_accumulators(target_codes, target_labels, K, b, ell)
    mu_t = rematerialize(S0_t, n_t, N_t, K, b, ell)

    target_conf = np.array([
        _qss_confidence(mu_t, all_codes[idx], int(labels[idx]), K, ell)
        for idx in audit_idx
    ])

    n_a = len(audit_idx)
    in_conf: list[list[float]] = [[] for _ in range(n_a)]
    out_conf: list[list[float]] = [[] for _ in range(n_a)]

    for s in range(n_shadow):
        shadow_mask = rng.rand(N) < 0.5
        for idx in audit_idx:
            shadow_mask[idx] = False
        shadow_idx = np.where(shadow_mask)[0]

        S0_s, n_s, N_s = build_accumulators(
            all_codes[shadow_idx], labels[shadow_idx], K, b, ell,
        )
        mu_s = rematerialize(S0_s, n_s, N_s, K, b, ell)

        for ai, idx in enumerate(audit_idx):
            conf_out = _qss_confidence(
                mu_s, all_codes[idx], int(labels[idx]), K, ell,
            )
            out_conf[ai].append(conf_out)

            # "IN" = shadow + this audit point. Increment accumulators
            # instead of rebuilding (O(ell^2 b^2) vs O(N)).
            S0_in = S0_s.copy()
            n_in = n_s.copy()
            N_in = {k: v.copy() for k, v in N_s.items()}
            c = all_codes[idx]
            lab = int(labels[idx])
            for lev in range(ell):
                S0_in[lev, c[lev], lab] += 1
                n_in[lev, c[lev]] += 1
            for m in range(ell):
                for k in range(m):
                    N_in[(k, m)][c[k], c[m]] += 1
            mu_in = rematerialize(S0_in, n_in, N_in, K, b, ell)
            conf_in = _qss_confidence(
                mu_in, all_codes[idx], int(labels[idx]), K, ell,
            )
            in_conf[ai].append(conf_in)

    lr_scores = np.zeros(n_a)
    for ai in range(n_a):
        in_vals = np.array(in_conf[ai])
        out_vals = np.array(out_conf[ai])
        if len(in_vals) < 2 or len(out_vals) < 2:
            continue
        mu_in_g = np.mean(in_vals)
        sigma_in = max(np.std(in_vals), 1e-10)
        mu_out_g = np.mean(out_vals)
        sigma_out = max(np.std(out_vals), 1e-10)
        tc = target_conf[ai]
        log_p_in = -0.5 * ((tc - mu_in_g) / sigma_in) ** 2 - np.log(sigma_in)
        log_p_out = (
            -0.5 * ((tc - mu_out_g) / sigma_out) ** 2 - np.log(sigma_out)
        )
        lr_scores[ai] = log_p_in - log_p_out

    auc = float(roc_auc_score(is_member.astype(int), lr_scores))

    tpr_at_fpr = {}
    n_neg = int((~is_member).sum())
    neg_scores = np.sort(lr_scores[~is_member])[::-1]
    for fpr_target in [0.001, 0.01, 0.1]:
        k = max(1, int(np.ceil(n_neg * fpr_target)))
        threshold = neg_scores[min(k - 1, len(neg_scores) - 1)]
        tpr = float(np.mean(lr_scores[is_member] >= threshold))
        tpr_at_fpr[f"TPR@FPR={fpr_target}"] = tpr

    return {
        "AUC": auc,
        **tpr_at_fpr,
        "n_members": int(is_member.sum()),
        "n_non_members": int((~is_member).sum()),
    }


# ── Codebook Fragility (LOO Flip Rate) ──


def codebook_fragility(
    embeddings: np.ndarray,
    b: int,
    ell: int,
    n_test: int = 200,
    seed: int = 42,
) -> dict:
    """Measure LOO code-flip rate: how often removing one training point
    from the codebook training set changes the RQ codes.

    For each of n_test points:
    1. Train RQ on full data, encode point
    2. Remove that point, retrain RQ, re-encode
    3. Check if any level's code flipped

    Also computes Voronoi margin statistics and dilution rho.

    Warning: retrains RQ n_test times. Use small n_test or small datasets.
    """
    import faiss

    rng = np.random.RandomState(seed)
    emb = l2_normalize(embeddings)
    N = len(emb)

    test_idx = rng.choice(N, min(n_test, N), replace=False)

    rq_full = train_rvq(emb, emb.shape[1], b, ell)
    codes_full = encode_rq(rq_full, emb, b, ell)

    flips = 0
    margins: list[float] = []

    for idx in test_idx:
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        rq_loo = train_rvq(emb[mask], emb.shape[1], b, ell)
        code_full = encode_rq(rq_full, emb[idx: idx + 1], b, ell)[0]
        code_loo = encode_rq(rq_loo, emb[idx: idx + 1], b, ell)[0]

        if not np.array_equal(code_full, code_loo):
            flips += 1

        codebooks = faiss.vector_to_array(rq_full.codebooks).reshape(
            ell, b, -1,
        )
        residual = emb[idx].astype(np.float64).copy()
        for level in range(ell):
            cb = codebooks[level]
            dists = np.sum((cb - residual) ** 2, axis=1)
            sorted_dists = np.sort(dists)
            margin = (
                sorted_dists[1] - sorted_dists[0]
                if len(sorted_dists) > 1
                else 0.0
            )
            margins.append(float(margin))
            assigned = int(np.argmin(dists))
            residual -= cb[assigned]

    flip_rate = flips / len(test_idx)

    # Dilution rho: measures code-level dependency
    dilutions: list[float] = []
    for m in range(ell):
        for k in range(m):
            joint = np.zeros((b, b), dtype=np.int64)
            np.add.at(joint, (codes_full[:, k], codes_full[:, m]), 1)
            row_sums = joint.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1)
            cond_probs = joint.astype(np.float64) / row_sums
            max_cond = cond_probs.max(axis=1)
            occupied = joint.sum(axis=1) > 0
            if occupied.any():
                dilutions.append(float(np.mean(max_cond[occupied])))

    rho = float(np.mean(dilutions)) if dilutions else 0.0

    return {
        "flip_rate": flip_rate,
        "n_test": len(test_idx),
        "margin_mean": float(np.mean(margins)),
        "margin_median": float(np.median(margins)),
        "margin_min": float(np.min(margins)),
        "margin_p01": float(np.percentile(margins, 1)),
        "dilution_rho": rho,
    }
