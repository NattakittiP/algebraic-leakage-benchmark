# Mathematical Appendix: Proof that TG4h Inclusion Reconstructs TCR

## Theorem

Including TG4h as a predictor, alongside TG0h, allows a machine-learning model to perfectly reconstruct the label `low_TCR` — regardless of the true predictive value of any other feature.

---

## Setup

Define the following quantities from the data-generating mechanism:

- **TG0h** : baseline triglyceride (mg/dL), available at time 0
- **TG4h** : post-heparin triglyceride (mg/dL), measured at 4 hours
- **TCR** : triglyceride clearance rate (%), defined as:

$$\text{TCR} = \frac{\text{TG0h} - \text{TG4h}}{\text{TG0h}} \times 100$$

- **low_TCR** : binary label, defined as:

$$\text{low\_TCR} = \mathbb{1}[\text{TCR} \leq Q_1(\text{TCR})]$$

where $Q_1$ is the 25th percentile of TCR in the training fold.

---

## Lemma 1: TG4h is a deterministic function of TG0h and TCR

Rearranging the TCR formula:

$$\text{TCR} = \frac{\text{TG0h} - \text{TG4h}}{\text{TG0h}} \times 100$$

$$\text{TG4h} = \text{TG0h} \times \left(1 - \frac{\text{TCR}}{100}\right)$$

Therefore TG4h is **fully determined** by TG0h and TCR. Given TG0h (available in the clean feature set), observing TG4h is equivalent to observing TCR:

$$\text{TCR} = 100 \times \left(1 - \frac{\text{TG4h}}{\text{TG0h}}\right)$$

---

## Theorem (Definitional Leakage)

**Claim:** A model with access to {clean features ∪ TG4h} can compute `low_TCR` exactly.

**Proof:**

Given any record where TG0h > 0 (guaranteed by the generator), any function $f$ with access to TG4h and TG0h can compute:

$$\hat{\text{TCR}} = 100 \times \left(1 - \frac{\text{TG4h}}{\text{TG0h}}\right) = \text{TCR} \quad \text{(exactly)}$$

Since the label is:

$$\text{low\_TCR} = \mathbb{1}[\hat{\text{TCR}} \leq Q_1(\text{TCR\_train})]$$

the model achieves **Bayes-optimal classification** with AUROC = 1.00 (subject only to threshold estimation error from the training fold).

In practice, the threshold is estimated from a finite training fold, so AUROC ≈ 0.95–1.00 rather than exactly 1.00. ∎

---

## Corollary 1: TCR as a Predictor

Including TCR directly achieves AUROC ≈ 1.00 trivially, since TCR IS the quantity used to derive the binary label. This is the most extreme leakage case and serves as the theoretical upper bound.

---

## Corollary 2: WBV Independence Under Null

In the null scenario, TCR is generated as:

$$\text{TCR}_i \sim \text{TruncNormal}(\mu_{TCR}, \sigma_{TCR}, -10, 99.9) \quad \text{independently of all covariates}$$

By construction, $\text{Cov}(\text{WBV}, \text{TCR}) = 0$ in the null scenario.

Therefore the clean AUROC for predicting `low_TCR` from WBV alone is:

$$\text{AUROC}(\text{WBV}, \text{low\_TCR}) = 0.50 \quad \text{(in expectation)}$$

The same holds for any function of the clean predictors: in the null scenario, no clean predictor provides information about `low_TCR`. ∎

---

## Implication for Researchers

Any published result with AUROC ≥ 0.70 in a paired triglyceride clearance task should be inspected for:

1. Inclusion of TG4h or equivalent post-heparin measurements as predictors
2. Inclusion of TCR or its equivalent as a predictor
3. Whether the label threshold was computed before or after train/test splitting

A rigorous benchmark must exclude all variables derived from the outcome.
