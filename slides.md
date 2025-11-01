---
marp: true
theme: default
class: lead
paginate: true
html: true
transition: slide
style: |
  img[title~="step"] {
    height: 64px;
    position: relative;
    top: -0.1em;
    vertical-align: middle;
    width: 64px;

    /* ⬇️ Mark the image of "1" in every pages as morphable image named as "one" ⬇️ */
    &[alt="3"] {
      view-transition-name: one;
    }

    img.with-transition[alt="3"] {
      view-transition-name: one;
    }

    img.no-transition[alt="3"] {
      view-transition-name: none;
    }
    p {
        font-size: 0.8em;
    }

    section li {
      font-size: 0.85em !important; /* force smaller size */
    }

  }

---

# Final Review Presentation
## Learning to Manipulate from Minimal Supervision
### Rhys Newbury


---

# Overview

This thesis explores methods for enabling robots to **learn and adapt their behaviour** in settings with **limited prior knowledge or supervision**, focusing on **manipulation tasks**.

The overarching goal is to develop **learning frameworks** that enable **efficient skill transfer** across **tasks** and **robot embodiments**, and for this we focus on ![3](https://icongr.am/material/numeric-3-circle.svg?color=FFC067 'step') research areas

---


# Research Focus

During this thesis, we tackle three core challenges:

- ![1](https://icongr.am/material/numeric-1-circle.svg?color=80EF80 'step') **Embodiment Transfer** — generalizing across robot morphologies ![](https://icongr.am/material/check-bold.svg?color=80EF80)
- ![2](https://icongr.am/material/numeric-2-circle.svg?color=80EF80 'step') **One-Shot Imitation** — transferring from a single demo ![](https://icongr.am/material/check-bold.svg?color=80EF80) 
- ![3](https://icongr.am/material/numeric-3-circle.svg?color=FFC067 'step') **Single-Demo Grasp Learning** — Learning grasping skills for a class of objects from <em>just one demonstration</em>.


All share a common theme:  
→ **Learning from limited data** while maintaining **broad generalization** across robots and tasks.

---

![3 w:256 h:256](https://icongr.am/material/numeric-3-circle.svg?color=FFC067 'step')

# Single-Demo Grasp Learning

---

# ![3](https://icongr.am/material/numeric-3-circle.svg?color=FFC067 'step') Single-Demo Grasp Learning


- Humans can generalize a grasp across different object instances easily:  
  e.g., always grasp a mug by the handle — regardless of shape variation.  
- Robots struggle with this: transferring a grasp from one object to another often needs retraining or dense data.  
- We aim to **enable grasp transfer** between novel objects using **only one demonstration**.



--- 
# Key Idea: Use 3D Keypoints as Semantic Anchors

<style scoped>
section li {
  font-size: smaller;
}
section p {
  font-size: small
}
</style>

- A grasp can be described by **where** contact happens, not just how the hand moves.  
- If we can learn **consistent 3D keypoints** across shapes (e.g., mug handles, bottle necks), then we can generalize a grasp demonstration to new objects from the same class.  
- These keypoints can serve as **semantic anchors** guiding grasp generation.


<div align="center">
  <img src="touchcode.png" alt="touchcode" width="70%">
</div>
<p>T. Zhu, R. Wu, J. Hang, X. Lin and Y. Sun, "Toward Human-Like Grasp: Functional Grasp by Dexterous Robotic Hand Via Object-Hand Semantic Representation," in IEEE Transactions on Pattern Analysis and Machine Intelligence</p>

---

<style scoped>
section li {
  font-size: smaller;
}
</style>


# Background

- 3D reconstruction methods typically compress geometric information into **latent vectors**.  
- Such embeddings, while effective, often lack **interpretability** — they don’t correspond to human-understandable object parts.  
- Structured representations (like **keypoints** or **skeletons**) provide a bridge: low-dimensional, intuitive, and semantically meaningful.  

<div align="center">
  <img src="latent.png" alt="latent" width="60%">
</div>

----

# High-Level Pipeline

<div style="text-align:center">
  <img src="https://example.com/network-diagram.png" width="750">
</div>

**Pipeline Overview**
1. **Input:** Single grasp demonstration on canonical object  
2. **Learned Keypoints:** Extract consistent 3D keypoints across shapes  
3. **Optimization:** Align robot fingertips to keypoints via differentiable simulation  


---

# Why Keypoints?

- Represent complex shapes in a **low-dimensional, interpretable** form  
- Enable **semantic transfer** (grasp-by-handle, not by absolute position)  
- Robust to noise and geometric variation  
- Learnable **without supervision** from 3D data:contentReference.

---

# Network Overview

We learn 3D keypoints from shapes using an **autoencoder-like** network:

Input Point Cloud (S₀)
↓
Encoder (PointTransformer)
↓
Structured Latent (Keypoints + Aux Code)
↓
Diffusion Decoder (Denoising)
↓
Reconstructed Shape


---

# Encoder

- Input: 3D point cloud \\( S_0 ∈ ℝ^{N×3} \\)
- Transformer-based encoder \\( F_θ \\) predicts attention weights \\( a_k \\)  
  to compute keypoints as convex combinations of input points:

\\[
K_k = \sum_i a_{ki} x_i, \quad \sum_i a_{ki} = 1, \; a_{ki} ≥ 0
\\]

- Ensures keypoints lie inside the convex hull of the object.

---

# Latent Representation

The encoder produces two outputs:
- **Keypoints (K):** structured, geometric features  
- **Auxiliary code (hₐₓ):** captures unstructured variation  

Combined latent vector:

\\[
z_0 = \text{vec}(K') \oplus z_{aux}
\\]

This structured latent feeds the decoder to guide shape generation:contentReference.

---

# Decoder (Diffusion-Based)

- Point-wise denoising network conditioned on \\( z_0 \\)
- Uses:
  1. Sinusoidal timestep embedding  
  2. Shape context vector from encoder  

- Cross-attention allows noisy points to align with global shape context  
- FiLM-conditioned MLP layers adaptively denoise points:contentReference.


---

# Training Objectives

| Term | Purpose |
|------|----------|
| \\( L_{diff} \\) | Denoising reconstruction |
| \\( L_{chamfer} \\) | Keypoint proximity to shape |
| \\( L_{mse} \\) | Deformation consistency |
| \\( L_{KL} \\) | Regularize auxiliary latent |
| \\( L_{FPS} \\) | Ensure keypoint coverage |

Together they align the latent space with the **ELBO objective**, ensuring both fidelity and interpretability:contentReference.

---

# Results

We evaluate our model on **ShapeNet** categories, comparing against keypoint-based and diffusion baselines (KPD, KeyGrid, SC3K, Skeleton Merger, and DPM).  
Performance is measured via:
- **Dual Alignment Score (DAS)** — keypoint quality  
- **Chamfer Distance (CD)** and **Earth Mover’s Distance (EMD)** — reconstruction quality  
- **MMD-CD / MMD-EMD** — generative quality:contentReference[oaicite:0]{index=0}.

---

# Quantitative Evaluation

| Model | DAS ↑ | CD ↓ | EMD ↓ | Notes |
|--------|-------|------|-------|-------|
| KPD | 0.72 | 0.015 | 1.5e−5 | baseline deformation model |
| DPM | 0.73 | 0.012 | 1.3e−5 | diffusion baseline |
| **Ours** | **0.78** | **0.0081** | **1.14e−5** | best correlation and fidelity |

- Our model achieves **~8% higher keypoint correlation** and **50% lower reconstruction error** than KeypointDeformer.  
- Consistent improvement across 16 ShapeNet categories:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}.

---

# Application: Grasp Transfer

1. Extract 3D keypoints on both canonical and target objects  
2. Compute correspondences between them  
3. Optimize the robot’s hand configuration to align fingertips with keypoints  
4. Refine grasp stability via differentiable physics simulation:contentReference.

---

# Summary

- **Problem:** Grasping from one demo → generalize across shapes  
- **Solution:** Learn unsupervised 3D keypoints as transferable anchors  
- **Approach:** Keypoint-conditioned latent diffusion model  
- **Outcome:** Physically stable and semantically consistent grasp transfer  

---

