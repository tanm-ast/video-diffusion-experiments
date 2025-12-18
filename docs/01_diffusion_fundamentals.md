# Diffusion Model Fundamentals

## 1. DDPM: Discrete Markov Chain

### Forward Process

Fixed corruption schedule over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

With $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$, you can skip to any timestep:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

Equivalently: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.

### Reverse Process

Learn to invert:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The training objective (simplified ELBO) reduces to noise prediction:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### Key Limitation

The reverse process is *stochastic*. Each step adds fresh noise scaled by $\sigma_t$. Skipping steps breaks the variance schedule and causes error accumulation.

---

## 2. SDE Formulation: Continuous Limit

As $T \to \infty$ with appropriate scaling, DDPM converges to a continuous-time SDE.

### Forward SDE (Variance Preserving)

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dw$$

where $w$ is a standard Wiener process and $\beta(t)$ is the continuous noise schedule.

This is an Ornstein-Uhlenbeck process. The drift term $-\frac{1}{2}\beta(t)x$ shrinks the signal, while $\sqrt{\beta(t)} dw$ adds noise.

### Reverse SDE (Anderson, 1982)

Any diffusion process with forward SDE $dx = f(x,t)dt + g(t)dw$ has a time-reversed SDE:

$$dx = \left[ f(x,t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) d\bar{w}$$

where $\bar{w}$ is a reverse-time Wiener process.

For VP-SDE:

$$dx = \left[ -\frac{1}{2}\beta(t)x - \beta(t) \nabla_x \log p_t(x) \right] dt + \sqrt{\beta(t)} \, d\bar{w}$$

The score $\nabla_x \log p_t(x)$ is what the network learns. Connection to noise prediction:

$$\nabla_x \log p_t(x) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

(Follows from Tweedie's formula applied to the Gaussian perturbation kernel.)

---

## 3. Probability Flow ODE: The Deterministic Twin

### Key Insight (Song et al., 2021)

For any SDE, there exists an ODE with *identical marginal distributions* $p_t(x)$ at all times.

For SDE: $dx = f(x,t)dt + g(t)dw$

The corresponding probability flow ODE is:

$$dx = \left[ f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right] dt$$

### Proof Sketch

The Fokker-Planck equation for the SDE describes how $p_t(x)$ evolves:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g^2 \nabla^2 p_t$$

The continuity equation for an ODE $dx = v(x,t)dt$ is:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (v p_t)$$

Setting these equal and solving for $v$:

$$v(x,t) = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

Both processes have identical densities at all times, but the ODE is deterministic—given $x_T$, the trajectory to $x_0$ is unique.

### For VP-SDE

The probability flow ODE is:

$$dx = -\frac{1}{2}\beta(t) \left[ x + \nabla_x \log p_t(x) \right] dt$$

Or in terms of the learned noise:

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t) \left[ x - \frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \right]$$

---

## 4. DDIM: Discretizing the ODE

DDIM (Song et al., 2020) derived the same result from a different angle—before the SDE framework was published. They showed you can define a family of non-Markovian forward processes that all have the same marginal $q(x_t|x_0)$ but different reverse processes.

### DDIM Update Rule

Given $x_t$, predict $x_0$:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

Then compute $x_{t-1}$:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$.

- When $\sigma_t = 0$: fully deterministic (probability flow ODE discretization)
- When $\sigma_t = \sqrt{(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t)} \sqrt{1-\bar{\alpha}_t/\bar{\alpha}_{t-1}}$: recovers DDPM

The deterministic case ($\sigma_t = 0$) allows arbitrary step skipping because you're just solving an ODE—no stochastic variance to track.

---

## 5. Euler Method and Modern Solvers

The probability flow ODE in the $\sigma$-parameterization (using $\sigma_t = \sqrt{(1-\bar{\alpha}_t)/\bar{\alpha}_t}$, i.e., signal-to-noise ratio):

$$\frac{dx}{d\sigma} = -\sigma \epsilon_\theta(x, \sigma)$$

### Euler Discretization

$$x_{\sigma_{i+1}} = x_{\sigma_i} + (\sigma_{i+1} - \sigma_i) \cdot \left( -\sigma_i \epsilon_\theta(x_{\sigma_i}, \sigma_i) \right)$$

This is first-order. Error scales as $O(\Delta \sigma)$.

### Higher-Order Solvers

**Heun (2nd order):** Predictor-corrector. Two function evaluations per step, error $O(\Delta \sigma^2)$.

**DPM-Solver (Lu et al.):** Exploits the semi-linear structure. The ODE can be written as:

$$\frac{dx}{d\lambda} = e^\lambda \epsilon_\theta(x, \lambda)$$

where $\lambda = \log \sigma$. Exact integration of the linear part + Taylor expansion of the nonlinear part gives multistep methods with fewer NFEs (neural function evaluations).

**DPM-Solver++:** Reformulates in terms of $\hat{x}_0$ prediction rather than $\epsilon$ prediction. More stable at low step counts.

### Solver Comparison

| Method | Order | Steps for Good Quality | NFE/Step |
|--------|-------|------------------------|----------|
| DDPM | - | 500-1000 | 1 |
| DDIM | 1 | 50-100 | 1 |
| Euler | 1 | 50-100 | 1 |
| Heun | 2 | 25-50 | 2 |
| DPM-Solver-2 | 2 | 20-30 | 1-2 |
| DPM-Solver++(2M) | 2 | 15-25 | 1 |
| UniPC | 2-3 | 10-20 | 1 |

---

## 6. Classifier-Free Guidance (CFG)

### The Problem

You want to generate images that match a text prompt. The diffusion model learns $p(x)$—the distribution of images. But you want $p(x|c)$—images *given* conditioning $c$ (your prompt).

### Original Classifier Guidance (Dhariwal et al.)

Train a separate classifier on noisy images. At each denoising step, compute the classifier's gradient with respect to the image and nudge the generation toward higher classifier probability for your target class. Works, but requires training a robust noisy-image classifier.

### Classifier-Free Guidance (Ho & Salimans)

During training, randomly drop the conditioning some percentage of the time (replace text embedding with null/empty). Now your single model learns *both*:
- $\epsilon_\theta(x_t, c)$ — noise prediction given conditioning
- $\epsilon_\theta(x_t, \varnothing)$ — noise prediction unconditionally

At inference, run the model twice per step and extrapolate:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, \varnothing) + s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)]$$

Where $s$ is the `guidance_scale`.

### Intuition

Compute the direction from "generic image" toward "image matching your prompt," then move in that direction with strength $s$.

### Effect of guidance_scale

| guidance_scale | Effect |
|----------------|--------|
| 1.0 | No guidance, just sample from learned distribution |
| 3-5 | Mild conditioning, more diverse/creative |
| 7-8 | Standard, balanced adherence |
| 10-15 | Strong adherence, risk of oversaturation |
| 20+ | Usually breaks—artifacts, blown-out colors |

### Relevance to Video

CFG scale interacts with temporal consistency. Higher CFG means stronger per-frame prompt adherence, which can *fight* the temporal attention that's trying to smooth across frames.

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
2. Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. ICLR.
3. Song, Y., et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR.
4. Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. NeurIPS Workshop.
5. Lu, C., et al. (2022). DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling. NeurIPS.
