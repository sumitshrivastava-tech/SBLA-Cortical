<div align="center">
  <h1>🧠 SBLA-Cortical (v0.9)</h1>
  <h3>Unifying 1.58-Bit Ternary Synapses, O(1) Dendritic Fast-Weight Memory, and Homeostatic Spiking Predictive Coding for Ultra-Efficient Foundation Models</h3>
  <p>
    <b>Invented, Architected & Developed by <a href="https://www.linkedin.com/in/sumitshrivastava-tech">Sumit Shrivastava</a></b><p>

  <p align="center">
  <a href="https://colab.research.google.com/github/sumitshrivastava-tech/SBLA-Cortical/blob/main/SBLA_Cortical_v09_Demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  &nbsp;&nbsp;
  <a href="https://doi.org/10.5281/zenodo.23499684">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23499684.svg" alt="DOI">
  </a>
</p>
    <a href="https://huggingface.co/SumitShrivastava/SBLA-Cortical-19M"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SBLA--Cortical--19M-blue" alt="Hugging Face"></a>
    <a href="./SBLA_Cortical_Sumit_Shrivastava_2026.pdf"><img src="https://img.shields.io/badge/Paper-PDF%20Monograph-b31b1b" alt="Paper PDF"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Academic%20Free%20%7C%20Commercial%20Royalty-green" alt="License"></a>
    <a href="https://www.linkedin.com/in/sumitshrivastava-tech"><img src="https://img.shields.io/badge/Author-Sumit%20Shrivastava-0077B5" alt="LinkedIn"></a>
  </p>
</div>

---

## ⚡ Executive Overview

**SBLA-Cortical** (*Sparse-Binary Liquid-Associative Cortical Architecture*) is a physics-grounded neuromorphic foundation model architected from mathematical first principles by **Sumit Shrivastava**.

Contemporary autoregressive Transformers suffer from four physical and algorithmic bottlenecks:
1. **The Von Neumann Memory Wall:** Hauling 16-bit floating-point parameters across off-chip DRAM/HBM consumes $500\times\text{–}833\times$ more energy than on-chip integer addition.
2. **Quadratic $\mathcal{O}(T^2)$ Time & Linear $\mathcal{O}(T)$ KV-Cache Growth:** Context memory explodes linearly with sequence length $T$.
3. **100% Synchronous Dense Activation:** Expending identical compute energy on trivial tokens (*"the"*, *"and"*) and complex reasoning alike.
4. **Unconstrained Euclidean Latent Spaces:** Allowing representation norm drift across recurrent reasoning steps.

**SBLA-Cortical (v0.9)** solves all four bottlenecks simultaneously by unifying:
* **Pillar I — Factorized 1.58-Bit Ternary Synaptic Algebra ($W \in \{-1, 0, +1\}$):** Compresses **81.1%** of network parameters (`15,361,856` core synapses) into 2-bit hardware-packed `uint8` integers (**4 synapses per byte**, occupying just **`3.64 MB`** of core weight RAM) paired with a continuous high-resolution sensory readout bridge ($d_s = 64$).
* **Pillar II — Zero-Quadratic $\mathcal{O}(1)$ Dual Cortical Working Memory (`72.0 KB` Constant State):** Eliminates the $T \times T$ attention matrix and growing KV-cache entirely via a **3-Tap Causal Dendritic Delay Line** fused with **Decoupled-RoPE Hebbian Fast-Weight Associative Memory** and a **Continuous-Time Liquid Time-Constant (LTC) Differential Equation**. Single-stream working memory remains strictly locked at **`72.0 KB`** across all context lengths ($1\text{K}$ to $131\text{K}+$ tokens).
* **Pillar III — Homeostatic Ternary LIF Spiking & Cortical Habituation:** Keeps **79.7% of cortical channels silent (`0`)** while automatically reducing active Mixture-of-Experts (MoE) routing from **69.8% down to 18.3%** as patterns transition from unfamiliar effort (*System 2*) to mastered habit (*System 1*).
* **Pillar IV — Monotonic 2-Step Hierarchical Predictive Coding on an RMS-Preserving Riemannian Manifold:** Contracts variational Free Energy by **6.0x** ($\mathcal{F}: 0.5676 \rightarrow 0.0943$) across two recurrent passes.

---

## 📊 Head-to-Head Benchmark: `SBLA-Cortical-v0.9` vs. Standard Dense Transformer

Evaluated side-by-side on real English BPE sequences (`tiktoken` GPT-2 vocabulary, $V = 50,257$, $L = 4, d_{\text{model}} = 256, H = 8$):

| Benchmark Metric | Standard Dense Transformer | **SBLA-Cortical-v0.9 (Ours)** | **SBLA-Cortical Advantage** |
| :--- | :--- | :--- | :--- |
| **Total Parameters** | 29,939,712 | **18,933,904 (~19M Class)** | **36.8% Fewer Parameters** |
| **Total Deployed Weight RAM** | 57.11 MB (FP16) | **10.48 MB** (`3.64 MB` Core + IO) | **5.4x Less Weight Memory** |
| **Core Reasoning Layer Weights** | 6.02 MB (FP16) | **3.64 MB (2-Bit Packed)** | **4 Synapses per Byte** |
| **Working Memory @ 1,024 Tokens ($B=1$)** | 4,096.0 KB (4.0 MB) | **72.0 KB** | **56.9x Less Working RAM** |
| **Working Memory @ 8,192 Tokens ($B=1$)** | 32,768.0 KB (32.0 MB) | **72.0 KB** | **455.1x Less Working RAM** |
| **Working Memory @ 32,768 Tokens ($B=1$)** | 131,072.0 KB (128.0 MB) | **72.0 KB** | **1,820.4x Less Working RAM** |
| **Working Memory @ 131,072 Tokens ($B=1$)** | 524,288.0 KB (512.0 MB) | **72.0 KB** | **7,281.8x Less Working RAM** |
| **Final Step-20 Active Neuron Firing** | 100.0% Dense Continuous | **20.3% Spiking (79.7% Silent)** | **4.9x Fewer Active Neurons** |
| **Final Step-20 Routed Expert Fraction** | 100.0% Dense FFN | **18.3% Active (81.7% Bypassed)** | **5.5x Less Expert Routing** |
| **Final Step-20 English CE Loss (PPL)** | `0.0080` (`1.01`) | **`0.0057` (`1.01`) [WINNER]** | **28.7% Lower Final Loss** |

---

## 🗂️ Repository Structure & Nine-Stage Ablation Suite (`v0.1` $\rightarrow$ `v0.9`)

```text
SBLA-Cortical/
├── modeling_sbla_cortical.py                  # Final verified SBLA-Cortical (v0.9) architecture
├── benchmark_v09_vs_transformer.py            # Reproducible head-to-head benchmark vs. Dense Transformer
├── SBLA_Cortical_Sumit_Shrivastava_2026.pdf   # Official 11-page Research Paper Monograph
├── requirements.txt                           # Python dependencies (PyTorch, tiktoken)
├── LICENSE                                    # SBLA-Cortical Academic & Commercial Royalty Dual License (v2.0)
└── ablations/                                 # Complete v0.1 -> v0.9 scientific discovery progression
    ├── README.md                              # Detailed ablation telemetry log
    ├── sbla_v01_initial_mla_liquid.py         # v0.1: Discovered Identity Residual Shortcut
    ├── sbla_v02_factorized_io_lif.py          # v0.2: Discovered SwiGLU Silent Cortex (0.9%) & Manifold Explosion
    ├── sbla_v03_rms_manifold_stateful_lif.py  # v0.3: RMS-Preserving Geodesic Manifold stabilization
    ├── sbla_v04_homeostatic_2bit_packer.py    # v0.4.1: Autograd-safe Homeostatic T-LIF & 3.64 MB 2-bit packer
    ├── sbla_v05_cortical_habituation_o1_gen.py# v0.5: Zero-KV O(1) autoregressive generation proof
    ├── sbla_v06_exact_causal_english_bpe.py   # v0.6: Step-exact scan; found repeated-word key aliasing
    ├── sbla_v07_dendritic_3tap_delay.py       # v0.7: 3-Tap Dendritic Delay cured key aliasing
    ├── sbla_v08_synaptic_brake_ablation.py    # v0.8: Identified subtracted-target Synaptic Brake bug
    └── sbla_v09_monotonic_pc_final.py         # v0.9: Monotonic Residual PC + Continuous Readout Bridge
```

---

## 🚀 Quickstart: Installation & Running the Benchmark

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/sumitshrivastava-tech/SBLA-Cortical.git
cd SBLA-Cortical
pip install -r requirements.txt
```

### 2. Run the Head-to-Head Benchmark (`SBLA-Cortical-v0.9` vs. Dense Transformer)
```bash
python modeling_sbla_cortical.py
```

### 3. Use `SBLA-Cortical` in Your Own Python Code
```python
import torch
from modeling_sbla_cortical import SBLACorticalLMV9, CorticalEnglishTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CorticalEnglishTokenizer(vocab_size=50257)

# Initialize SBLA-Cortical-19M (v0.9) and pack 15.36M ternary synapses into 3.64 MB (2-bit uint8)
model = SBLACorticalLMV9(vocab_size=50257).to(device)
packed_mb = model.compress_ternary_core_2bit()
print(f"2-Bit Packed Ternary Core RAM: {packed_mb} MB")

# Generate English text using ONLY 72.0 KB of O(1) Working Memory (Zero KV-Cache!)
prompt_ids = torch.tensor([tokenizer.encode("Once upon a time, a clever little robot")], device=device)
output_ids, state_kb = model.generate_o1_state(prompt_ids, max_new_tokens=16)
print("Generated:", tokenizer.decode(output_ids[0].tolist()))
print(f"Constant O(1) Working Memory Used: {state_kb} KB")
```

---

## ⚖️ License: Academic Free | Commercial Royalty-Bearing (v2.0)

The **SBLA-Cortical** architecture, source code, mathematical formulations, and model weights are the sole intellectual property of **Sumit Shrivastava** and are released under the **SBLA-Cortical Academic & Commercial Royalty Dual License (v2.0)** (see [`LICENSE`](./LICENSE)):

* **🎓 100% Free for Students & Non-Commercial Academic Researchers:** Individual students, independent learners, and academic researchers affiliated with accredited educational institutions are granted a free license to study, execute, fine-tune, and publish non-commercial academic benchmarks with proper citation to **Sumit Shrivastava (2026)**.
* **🏢 Mandatory Royalty & Equity License for Commercial Entities:** Any commercial use, integration into revenue-generating software, cloud API serving, hardware/edge deployment, or use inside a for-profit startup or enterprise is **STRICTLY PROHIBITED** without a prior written Commercial Royalty & Equity License Agreement with **Sumit Shrivastava** (granting a negotiated 5%–15% gross revenue royalty and/or 5%–10% equity share).

---

## 📚 Citation

If you use, reference, or build upon **SBLA-Cortical** in any research paper, repository, or benchmark, please cite:

```bibtex
@article{shrivastava2026sblacortical,
  title   = {SBLA-Cortical: Unifying 1.58-Bit Ternary Synapses, O(1) Dendritic Fast-Weight Memory, and Homeostatic Spiking Predictive Coding for Ultra-Efficient Foundation Models},
  author  = {Shrivastava, Sumit},
  year    = {2026},
  doi     = {10.5281/zenodo.23499684},
  url     = {https://doi.org/10.5281/zenodo.23499684},
  note    = {GitHub: https://github.com/sumitshrivastava-tech/SBLA-Cortical | Hugging Face: https://huggingface.co/SumitShrivastava/SBLA-Cortical-19M}
}
```
