Use transfer learning
Use vision transformers
Stack them for ensemble training if needed
Use CNNs
Study weekly quiz for exam

Use bayesian optimizer (optuna) for parameter search space


Don't use Relu, use PReLu or something else
Dying relu -> use Leaky ReLU/ELU/GELU, apply He initialization, and control learning rates.
Don't use normalize (0.5,0.5), it will kill all the pixels in mnist to black


https://d2l.ai/chapter_preliminaries/index.html