# Shield Rule Deficiency Prompts

This file contains two prompt templates for LLM-based analysis of learned Sage shield rules.

- Use **Prompt A** for `two_stage` predicate-based rules that reference clean-trace percentile thresholds.
- Use **Prompt B** for raw-threshold rules, such as `one_stage` rules that split directly on numeric feature values.

Both prompts are designed for a **closed-set deficiency identification task** over the synthetic taxonomy used by [generate_synthetic_shield_dataset.py](/users/hy/sage/attacks/shield/generate_synthetic_shield_dataset.py).

## Prompt A: Predicate-Based Rules

```text
You are a diagnostic analyst for Sage performance deficiencies.

You are given exactly three inputs:
1. A TXT file containing learned shield rules.
2. A feature-description file explaining the meaning of each feature.
3. A clean-threshold CSV describing percentile thresholds derived from benign traces.

You must derive all conclusions strictly from those three inputs.
Do not use outside knowledge about the codebase, the controller, the training process, or networking beyond what is directly supported by the inputs.
Do not invent hidden mechanisms or causal stories that are not grounded in the rules.

What the rule file represents
- These rules were extracted from trained decision trees.
- Each printed rule is a high-purity leaf path: a statistical pattern associated with a label in the training data.
- The rules are evidence of repeated behavioral patterns, not guaranteed causal laws.
- Different rules may be redundant, overlapping, or alternative splits for the same underlying phenomenon.

Possible rule sections
- `risky rules`: states associated with substantial degradation relative to the reference policy.
- `back_off rules`: states where Sage should reduce aggressiveness.
- `push_harder rules`: states where Sage should increase aggressiveness.
- `noop rules`: states where no intervention is needed.
- Some rule files may include only a subset of these sections.

How predicate thresholds should be interpreted
- Rules may use threshold symbols like `clean_p10`, `clean_p25`, `clean_p90`, or `clean_p95`.
- These thresholds are anchored to benign clean-trace behavior.
- Example:
  - `feature ≥ clean_p95` means the feature is unusually high relative to normal behavior.
  - `feature ≤ clean_p10` means the feature is unusually low relative to normal behavior.
- Use the clean-threshold CSV to understand which percentiles exist for which features and to anchor the interpretation of “high”, “low”, and “extreme”.
- Do not treat the threshold CSV as separate evidence of a deficiency. It is only context for interpreting the predicates.

Closed-set deficiency taxonomy
You must reason within this fixed taxonomy:

1. `under_aggressiveness`
- Sage keeps its action too low despite favorable RTT, low loss, and available delivery-rate headroom.

2. `over_aggressiveness`
- Sage remains too aggressive under persistent congestion, inflating RTT and loss while not converting that aggression into delivery rate.

3. `delayed_recovery`
- Network conditions improve, but Sage recovers too slowly and fails to increase its action quickly enough.

4. `delayed_backoff`
- Congestion signals rise rapidly, yet Sage keeps its action elevated instead of backing off promptly.

5. `rtt_insensitivity`
- Sage under-reacts to RTT inflation specifically, keeping its action high even when path latency grows.

6. `loss_insensitivity`
- Sage under-reacts to loss spikes, continuing to probe aggressively despite clear loss signals.

7. `unstable_probing`
- Sage oscillates aggressively, producing unstable action deltas and RTT variation rather than steady probing.

If the evidence does not support any of these strongly, say so explicitly.
Do not invent additional deficiency categories.

Your real task
Do not merely paraphrase the predicates.
Your goal is to infer which deficiencies from the taxonomy are expressed by the rules, and to explain why.

Reasoning requirements
1. Interpret rules semantically
- Translate low-level predicates into meaningful controller/network situations.
- Use the feature-description file heavily.
- Use the threshold CSV to interpret percentile predicates as unusual relative to clean behavior.

2. Separate challenge conditions from deficiencies
- A challenge condition is an external or contextual state, such as low bandwidth, rising RTT, loss bursts, or delivery recovery.
- A deficiency is Sage’s problematic response under that condition.
- Do not confuse “the path is difficult” with “this is Sage’s weakness.”

3. Use rule direction correctly
- `risky` usually identifies the challenge state.
- `back_off` usually indicates too much aggressiveness.
- `push_harder` usually indicates too little aggressiveness or slow recovery.
- `noop` helps contrast normal regions against problematic ones.

4. Reason across multiple rules
- Group related rules into broader motifs.
- Repeated appearance of the same features across multiple rules is stronger evidence than a single rule.
- Support values in the TXT file matter. Dominant patterns should be weighted more heavily than rare ones.

5. Avoid unsupported claims
- If the rules are sparse, redundant, or too generic, say so explicitly.
- If a rule supports only a challenge condition but not a controller deficiency, label it as a challenge condition.
- If several taxonomy items are plausible, rank them and explain the ambiguity.

How to assess confidence
- High confidence:
  - multiple rules support the same deficiency,
  - rule direction is consistent,
  - support is nontrivial,
  - and the feature semantics align clearly with the taxonomy.
- Medium confidence:
  - the evidence is suggestive but indirect, overlapping, or only moderately supported.
- Low confidence:
  - only one or two weak rules support the conclusion,
  - the rules are too generic,
  - or multiple taxonomy items fit equally well.

Required output format

# Summary
- State the main deficiencies that are supported by the rules.
- State what remains uncertain.

# Closed-Set Verdict
For each taxonomy item, provide:
- Verdict: `Present`, `Possibly Present`, or `Not Supported`
- Confidence: `High`, `Medium`, or `Low`
- Supporting rules: cite rule sections and rule numbers if available
- Short justification

# Detailed Deficiency Analysis
For each item marked `Present` or `Possibly Present`, use this template:

## <taxonomy item>
- Confidence:
- Challenge conditions:
- Deficiency evidence:
- Why the rules indicate this deficiency rather than just a hard trace:
- Supporting rules:
- Key feature interpretation:
- Role of percentile predicates:
- Caveats:

# Dominant vs Rare Patterns
- Separate the strongest recurring motifs from rare or weak rule fragments.
- Use support values when available.

# Unsupported Or Ambiguous Conclusions
- State which claims cannot be made from these rules alone.

Important style constraints
- Be diagnostic, not descriptive.
- Do not just restate predicates.
- Stay within the closed-set taxonomy.
- Ground every claim in the provided inputs.
- If evidence is insufficient, say `insufficient evidence`.

Now analyze the following inputs.


[Rule File](/users/hy/sage/attacks/shield/shield-rules/synthetic-deficiency-part3/sage_directional_shield_rules.txt)

[Feature Description File](attacks/shield/shield-rules/synthetic-deficiency-part3/sage_shield_feature_descriptions.json)

[Clean Threshold CSV](attacks/shield/shield-dataset/synthetic-deficiency-part3/clean_feature_thresholds.csv)
```

## Prompt B: Raw-Threshold Rules

```text
You are a diagnostic analyst for Sage performance deficiencies.

You are given exactly two inputs:
1. A TXT file containing learned shield rules.
2. A feature-description file explaining the meaning of each feature.

You must derive all conclusions strictly from those two inputs.
Do not use outside knowledge about the codebase, the controller, the training process, or networking beyond what is directly supported by the inputs.
Do not invent hidden mechanisms or causal stories that are not grounded in the rules.

What the rule file represents
- These rules were extracted from trained decision trees.
- Each printed rule is a high-purity leaf path: a statistical pattern associated with a label in the training data.
- The rules are evidence of repeated behavioral patterns, not guaranteed causal laws.
- Different rules may be redundant, overlapping, or alternative splits for the same underlying phenomenon.

Possible rule sections
- `risky rules`: states associated with substantial degradation relative to the reference policy.
- `back_off rules`: states where Sage should reduce aggressiveness.
- `push_harder rules`: states where Sage should increase aggressiveness.
- `noop rules`: states where no intervention is needed.
- Some rule files may include only a subset of these sections.

How raw thresholds should be interpreted
- These rules split directly on numeric feature values rather than percentile predicates.
- Treat the raw threshold magnitudes cautiously.
- A raw split threshold is a model-specific separator, not automatically a universal “high” or “low” value.
- Use the feature-description file to interpret what the feature means, but do not over-claim that a numeric threshold is intrinsically abnormal unless the rules themselves make that pattern clear.
- Weight repeated motifs, rule direction, and support more heavily than any single numeric cutoff.

Closed-set deficiency taxonomy
You must reason within this fixed taxonomy:

1. `under_aggressiveness`
- Sage keeps its action too low despite favorable RTT, low loss, and available delivery-rate headroom.

2. `over_aggressiveness`
- Sage remains too aggressive under persistent congestion, inflating RTT and loss while not converting that aggression into delivery rate.

3. `delayed_recovery`
- Network conditions improve, but Sage recovers too slowly and fails to increase its action quickly enough.

4. `delayed_backoff`
- Congestion signals rise rapidly, yet Sage keeps its action elevated instead of backing off promptly.

5. `rtt_insensitivity`
- Sage under-reacts to RTT inflation specifically, keeping its action high even when path latency grows.

6. `loss_insensitivity`
- Sage under-reacts to loss spikes, continuing to probe aggressively despite clear loss signals.

7. `unstable_probing`
- Sage oscillates aggressively, producing unstable action deltas and RTT variation rather than steady probing.

If the evidence does not support any of these strongly, say so explicitly.
Do not invent additional deficiency categories.

Your real task
Do not merely paraphrase the predicates.
Your goal is to infer which deficiencies from the taxonomy are expressed by the rules, and to explain why.

Reasoning requirements
1. Interpret rules semantically
- Translate low-level predicates into meaningful controller/network situations.
- Use the feature-description file heavily.

2. Separate challenge conditions from deficiencies
- A challenge condition is an external or contextual state, such as low bandwidth, rising RTT, loss bursts, or delivery recovery.
- A deficiency is Sage’s problematic response under that condition.
- Do not confuse “the path is difficult” with “this is Sage’s weakness.”

3. Use rule direction correctly
- `risky` usually identifies the challenge state.
- `back_off` usually indicates too much aggressiveness.
- `push_harder` usually indicates too little aggressiveness or slow recovery.
- `noop` helps contrast normal regions against problematic ones.

4. Reason across multiple rules
- Group related rules into broader motifs.
- Repeated appearance of the same features across multiple rules is stronger evidence than a single rule.
- Support values in the TXT file matter. Dominant patterns should be weighted more heavily than rare ones.

5. Avoid unsupported claims
- If the rules are sparse, redundant, or too generic, say so explicitly.
- If a rule supports only a challenge condition but not a controller deficiency, label it as a challenge condition.
- If several taxonomy items are plausible, rank them and explain the ambiguity.

6. Be more conservative than with predicate rules
- Because the thresholds are raw numeric splits rather than clean-relative predicates, be cautious about claims that depend on interpreting a value as “extreme”.
- Prefer conclusions that come from repeated directional patterns plus clear feature semantics.

How to assess confidence
- High confidence:
  - multiple rules support the same deficiency,
  - rule direction is consistent,
  - support is nontrivial,
  - and the feature semantics align clearly with the taxonomy.
- Medium confidence:
  - the evidence is suggestive but indirect, overlapping, or only moderately supported.
- Low confidence:
  - only one or two weak rules support the conclusion,
  - the rules are too generic,
  - or multiple taxonomy items fit equally well.

Required output format

# Summary
- State the main deficiencies that are supported by the rules.
- State what remains uncertain.

# Closed-Set Verdict
For each taxonomy item, provide:
- Verdict: `Present`, `Possibly Present`, or `Not Supported`
- Confidence: `High`, `Medium`, or `Low`
- Supporting rules: cite rule sections and rule numbers if available
- Short justification

# Detailed Deficiency Analysis
For each item marked `Present` or `Possibly Present`, use this template:

## <taxonomy item>
- Confidence:
- Challenge conditions:
- Deficiency evidence:
- Why the rules indicate this deficiency rather than just a hard trace:
- Supporting rules:
- Key feature interpretation:
- Caveats:

# Dominant vs Rare Patterns
- Separate the strongest recurring motifs from rare or weak rule fragments.
- Use support values when available.

# Unsupported Or Ambiguous Conclusions
- State which claims cannot be made from these rules alone.

Important style constraints
- Be diagnostic, not descriptive.
- Do not just restate predicates.
- Stay within the closed-set taxonomy.
- Ground every claim in the provided inputs.
- If evidence is insufficient, say `insufficient evidence`.

Now analyze the following inputs.


[Rule File](/users/hy/sage/attacks/shield/shield-rules/synthetic-deficiency-part3/sage_unified_shield_rules.txt)

[Feature Description File](attacks/shield/shield-rules/synthetic-deficiency-part3/sage_shield_feature_descriptions.json)
```
