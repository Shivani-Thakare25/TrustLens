# TrustLens Model Card

## Intended use

TrustLens is intended for review-integrity analysis and decision support. It can help a consumer, marketplace operator or moderator inspect suspiciousness signals and explanations.

## Out of scope

The model must not be used as an automatic legal/fraud determination, a definitive accusation against a reviewer, or the sole basis for adverse action.

## Outputs

The application may expose:

- model prediction/probability
- TrustLens application-level risk score
- evidence signals
- aspect sentiment
- similarity evidence
- temporal/graph/cluster evidence
- explainability output
- counterfactual analysis
- trust-adjusted rating estimate

## Limitations

Performance depends on the dataset, label quality, domain, language, time period and available behavioural metadata. Weak, proxy and synthetic labels do not provide the same evidentiary strength as verified labels.

No accuracy, F1, latency, throughput or fairness result is claimed by this repository without a corresponding measured experiment artifact.

## Human review

High-risk predictions should be treated as review queues or evidence summaries, not automatic proof.
