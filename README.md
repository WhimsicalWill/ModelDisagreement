# Model Disagreement

Using an ensemble of learned dynamics models, we construct a disagreement loss that guides the agent to explore novel parts of the environment and resolve epistemic uncertainty.

# TODO

- 1) Collect random experience from the environment
- 2) Learn ensemble of world models using experience (for a fixed number of steps)
- 3) Collect experience using Actor policy and update params after every step