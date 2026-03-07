## Project Objective
The goal of this project is to first systematically identify challenging network environments and conditions that expose failure cases in RL-based network controllers, such as adaptive bitrate streaming, congestion control, and resource scheduling. Specifically, we aim to find scenarios where the agent’s performance is significantly suboptimal compared to the theoretical or approximated optimal value. We don't assume any specific attack model or environment perturbation, but instead take a broad approach to explore various types of challenging conditions, including but not limited to adversarial attacks, distribution shifts, and rare edge cases.

We then analyze these failure cases to extract insights about the model’s weaknesses and use them to improve the controller, for example, through targeted fine-tuning on challenging environments and/or incorporating formal logic mechanisms such as "shields" as guardrails.

## Python Comment Style

- Use comment prefixes such as `#*`, `#^`, and `#&` for highlighting.
- For commented out code, just use `#` without any special prefix.