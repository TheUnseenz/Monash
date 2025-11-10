## Page 1: Foundations, Optimization, CNNs, & Adversarial

**1. Fundamentals: Overfitting & Gradients**
* **Overfit:** High train acc, low val acc. **Fix:** Data Aug, L2, Dropout, Early Stop.
* **Underfit:** Low train acc, low val acc. **Fix:** Bigger model, train longer.
* **Vanishing Grad:** $\nabla \to 0$. **Fix:** ReLU, ResNet (Skip Connects), Batch Norm, He Init.
* **Exploding Grad:** $\nabla \to \infty$. **Fix:** Gradient Clipping, ResNet.
* **Total Loss:** $J(\theta) = \text{Empirical Loss}$ (e.g., Cross-Entropy) + $\text{Regularization}$ (e.g., L2).
* **L2:** $\Omega(\theta) = \lambda \sum ||W^k||^2_F$ (small weights). **L1:** $\Omega(\theta) = \lambda \sum |W^k|$ (sparse weights).

**2. Optimization & Backpropagation**
* **Critical Point:** $\nabla_\theta J(\theta) = 0$. Can be Local Min, Max, or **Saddle Point**.
* **SGD (Stochastic Gradient Descent):** $\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{b}\sum_{k=1}^{b} \nabla_{\theta}l_k$. (Adam is common default).
* **Key Gradients (Backprop):**
    * **CE+Softmax:** $\frac{\partial l}{\partial h} = p - 1_y$ (where $h$ is logits, $p$ is softmax output, $1_y$ is one-hot label).
    * **Activation ($h' = \sigma(h)$):** $\frac{\partial l}{\partial \mathbf{h}} = \frac{\partial l}{\partial \mathbf{h'}} \odot \sigma'(\mathbf{h})$.
        * **ReLU Derivative:** $\sigma'(x) = 1$ if $x > 0$, $0$ if $x \le 0$.
    * **Linear ($h = xW+b$):** $\frac{\partial l}{\partial \mathbf{W}} = \mathbf{x}^T \frac{\partial l}{\partial \mathbf{h}}$ (Outer product).

**3. Core Techniques**
* **Initialization:** **He** for `ReLU`, **Xavier/Glorot** for `tanh`.
* **Batch Norm (BN):** Normalizes layer inputs *per batch*: $\hat{x} = (x - \mu_B) / \sigma_B$. Then learns affine params: $y = \gamma \hat{x} + \beta$. Stabilizes training, acts as regularizer.
* **Dropout:** **Train time:** randomly set $p$ fraction of neurons to 0. **Test time:** use all neurons. Prevents co-adaptation.
* **Data Augmentation:** Create synthetic training data (e.g., flip, crop, color shift).

**4. Convolutional Neural Networks (CNNs)**
* **Architecture:** `[CONV -> ReLU -> POOL]` $\times N \to \text{FLATTEN} \to \text{FC} \to \text{SOFTMAX}$.
* **Conv Output Size:** $W_{out} = \lfloor (W_{in} - F + 2P)/S \rfloor + 1$. (W=Width, F=Filter, P=Pad, S=Stride).
* **Receptive Field:** Input region "seen" by a neuron. Grows in deeper layers.
* **Global Average Pooling (GAP):** Replaces `Flatten`. Reduces `[C, H, W]` $\to$ `[C]` vector by averaging each channel. Reduces params.
* **ResNet (Residual Networks):**
    * **Core:** $h(x) = \text{ReLU}(\text{MainPath}(x) + x)$.
    * **Skip Connection:** The `+ x` path allows gradients to flow unimpeded, fixing vanishing gradient.
    * **1x1 Conv:** Used in skip connection *only if* dimensions change (e.g., `stride=2` or different # channels).

**5. Adversarial Attacks**
* **Concept:** $x_{adv} = x + \delta$ (small, imperceptible perturbation $\delta$).
* **Constraint:** $||x_{adv} - x||_\infty \le \epsilon$ (max change to any single pixel is $\epsilon$).
* **Untargeted:** $\max_{x'} l(f(x'), y_{\text{true}})$. Goal: Cause *any* misclassification.
    * **FGSM:** $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x l(f(x), y))$.
* **Targeted:** $\min_{x'} l(f(x'), y_{\text{target}})$. Goal: Cause *specific* misclassification.
* **Defense (Adversarial Training):** Train model on a mix of clean $x$ and $x_{adv}$.

---

## Page 2: Sequences, Transformers, & Generative Models

**6. Recurrent Neural Networks (RNNs)**
* **Core:** $h_t = f(h_{t-1}, x_t)$. Has a "loop" to process sequences. Fails at long-term dependencies (vanishing/exploding grads).
* **LSTM (Long Short-Term Memory):** Solves this with:
    * **Cell State ($c_t$):** "Long-term memory" conveyor belt.
    * **Hidden State ($h_t$):** "Short-term memory" / output.
    * **Gates (Sigmoid 0-1):** **Forget ($f_t$)** controls $c_{t-1}$; **Input ($i_t$)** controls new info $\tilde{c}_t$; **Output ($o_t$)** controls $h_t$.
    * **Updates:** $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$; $h_t = o_t \odot \tanh(c_t)$.
* **Topologies:** Many-1 (Sentiment), One-Many (Caption), Many-Many (Translate).

**7. Word Embeddings (Word2Vec)**
* **Goal:** Learn semantic word vectors ($v_{\text{King}} - v_{\text{Man}} + v_{\text{Woman}} \approx v_{\text{Queen}}$).
* **CBOW:** `Context words -> Target word`.
* **Skip-Gram:** `Target word -> Context words`.
* **Negative Sampling:** Replaces softmax. Binary task: `(word, context)` real (1) or fake (0)?

**8. Seq2Seq & Attention**
* **Encoder-Decoder (Seq2Seq):**
    1.  **Encoder** RNN reads input $\to$ fixed-size **Context Vector $c$** (final $h_T$).
    2.  **Decoder** RNN uses $c$ to generate output.
    * **Problem:** $c$ is a bottleneck for long sequences.
* **Attention:** Fixes bottleneck. Decoder "looks back" at all encoder states $h_i$.
    * **Dynamic Context:** $c_j = \sum_i \alpha_{ji} h_i$, where $\alpha_{ji} = \text{softmax}(\text{score}(q_j, h_i))$.
    * $q_j$ is decoder state (Query), $h_i$ are encoder states (Keys/Values).

**9. Transformers (Attention is All You Need)**
* **Core:** No RNNs. Parallel. Uses **Positional Encoding** (sin/cos) for order info.
* **Scaled Dot-Product Self-Attention:**
    * `Q`, `K`, `V` from *same* sequence (e.g., $Q=XW_Q, K=XW_K, V=XW_V$).
    * $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V$
* **Multi-Head (MHA):** Runs $h$ heads (w/ different $W_Q, W_K, W_V$) in parallel, concatenates results.
* **Transformer Block:** `[MHA -> Add & Norm -> FFN -> Add & Norm]`
    * **Add & Norm:** Residual Connection + **Layer Norm**.
    * **Layer Norm:** Normalizes across *feature* dim (vs. *batch* dim in BN).
* **Cross-Attention:** In Decoder. `Q` from Decoder, `K` & `V` from Encoder output.

**10. Vision Transformer (ViT)**
* **Process:** Image $\to$ 16x16 **Patches** ("tokens") $\to$ Linear Embed $\to$ + `[CLS]` token $\to$ + Pos. Encoding $\to$ Transformer Encoder $\to$ Classify with `[CLS]` output.
* **Data-Hungry:** Lacks CNN inductive bias, needs *massive* pre-training.
* **Swin-T:** Hierarchical ViT. Uses **Windowed (W-MSA)** & **Shifted (SW-MSA)** attention for efficiency.

**11. PEFT (Parameter-Efficient Fine-Tuning)**
* **Goal:** Freeze pre-trained model, train few new params.
* **LoRA (Low-Rank Adaptation):** $W_{new} = W_{frozen} + B \cdot A$. Freezes $W$, trains small $B, A$.
* **Adapters:** Small FFNs added *inside* Transformer blocks.
* **Prompt Tuning:** Learnable "prompt" vectors added to the *input*.

**12. Generative Models**
* **GANs (Generative Adversarial Networks):**
    * **G (Generator/Counterfeiter):** $z \text{ (noise)} \to G(z) \text{ (fake image)}$.
    * **D (Discriminator/Police):** $x \text{ (real/fake)} \to \text{0 (fake) or 1 (real)}$.
    * **Minimax Game:** $\min_G \max_D \mathbb{E}_{x \sim p_d}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$.
    * **Nash Equilibrium:** $p_g = p_d$ (fakes are perfect) & $D(x) = 0.5$ ($D$ is confused).
    * **Mode Collapse:** $G$ only generates a few types of fakes.
* **Diffusion Models:**
    * **Forward (Fixed):** $x_0 \to \dots \to x_T$ (Gradually add noise $\epsilon$).
    * **Reverse (Learned):** $x_T \to \dots \to x_0$ (Gradually denoise). Train **U-Net** $\epsilon_\theta(x_t, t)$ to *predict the noise $\epsilon$* that was added.
    * **Loss:** $\min ||\epsilon_\theta(x_t, t) - \epsilon||^2$ (predict the added noise).