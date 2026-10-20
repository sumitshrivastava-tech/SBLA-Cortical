# 🔬 SBLA-Cortical Evolutionary Ablation Suite (`v0.1` $\rightarrow$ `v0.9`)

**Author:** Sumit Shrivastava

This directory contains the complete nine-stage evolutionary progression of **SBLA-Cortical**, documenting the five novel neuromorphic failure modes discovered and mathematically resolved during the research monograph.

| File | Version | Key Scientific Discovery / Failure Mode Resolved |
| :--- | :---: | :--- |
| `sbla_v01_initial_mla_liquid.py` | **`v0.1`** | **Discovery 1 (Identity Residual Shortcut):** Identified how unshifted targets + unscaled tied embeddings allowed input $x_t$ to echo directly to logits (`0.0021` leak). |
| `sbla_v02_factorized_io_lif.py` | **`v0.2`** | **Discovery 2 (Silent Cortex & Manifold Explosion):** Factorized Sensory I/O saved 9.6M FP32 params (`18.67M` total); discovered SwiGLU variance collapse (`0.9%` spikes) and $L_2$ Free-Energy explosion (`1.2345`). |
| `sbla_v03_rms_manifold_stateful_lif.py` | **`v0.3`** | **Discovery 3 (RMS Geodesic Stability):** Introduced RMS-preserving Riemannian projection (`FE: 0.2639 -> 0.1383`) and stateful LIF integration. |
| `sbla_v04_homeostatic_2bit_packer.py` | **`v0.4.1`** | **Autograd-Safe Homeostasis & 2-Bit Packing:** Cured PyTorch in-place autograd version crash (`v8 vs v6`) via `.detach().clone()` snapshots and packed 15.36M ternary weights into `3.64 MB`. |
| `sbla_v05_cortical_habituation_o1_gen.py` | **`v0.5`** | **Zero-KV $\mathcal{O}(1)$ Generation:** Proved 100% exact sequence continuation from `68.0 KB` state with zero KV-cache. |
| `sbla_v06_exact_causal_english_bpe.py` | **`v0.6`** | **Discovery 4 (Associative Key Aliasing):** Verified `0.000098` state drift on real English `tiktoken`; discovered repeated word *"the"* aliased *"explored the birds"*. |
| `sbla_v07_dendritic_3tap_delay.py` | **`v0.7`** | **Cured Key Aliasing:** Added 3-Tap Causal Dendritic Delay Line (`72.0 KB` state), disambiguating *"explored the forest"* vs. *"helped the birds"*. |
| `sbla_v08_synaptic_brake_ablation.py` | **`v0.8`** | **Discovery 5 (Synaptic Brake Bug):** Demonstrated how subtracting unanchored target prediction error fought MoE updates (`3.0718` loss). |
| `sbla_v09_monotonic_pc_final.py` | **`v0.9`** | **Final Breakthrough:** Monotonic 2-Step Residual Predictive Coding + Continuous Readout Bridge achieved **`0.0057` CE Loss**, beating the Standard Dense Transformer (`0.0080`) with **6.0x** Free-Energy contraction. |
