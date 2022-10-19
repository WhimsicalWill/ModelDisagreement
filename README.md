# Model Disagreement

Using an ensemble of learned dynamics models, we construct a disagreement loss that guides the agent to explore novel parts of the environment and resolve epistemic uncertainty.

# TODO

- At each step before executing an action, optimize the actor parameters w.r.t. the intrinsic loss of an n-step trajectory. We can linearly expand this horizon during training