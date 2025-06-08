---
layout: default
title: Optimal Transport in Generative Modeling --an Introduction
date: 2025-01-31 01:00:39
collection: blogs
mathjax: true
hide_github_button: true
---


<head>
  <!-- MathJax v3 + SVG 渲染 + 自定义配置 -->
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        tags: 'ams'
      },
      svg: {
        fontCache: 'global',
        scale: 1.1 // 可选：放大字体，美观一些
      }
    };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>



*Note: As the author's research focuses on generative models in AI rather than pure mathematics, rigorous mathematical proofs and prerequisite knowledge are omitted. References are temporarily excluded.*

Cover image from: [ICERM Workshop on Optimal Transport](https://icerm.brown.edu/program/topical_workshop/tw-23-otds)

## 1. Introduction

Optimal Transport (OT) theory originated from Gaspard Monge's 18th-century transportation problem, which sought to move mass (e.g., earth) from one location to another at "minimal cost." In the 1940s, Leonid Kantorovich proposed a relaxed formulation, establishing the modern OT framework. This article introduces OT's foundational concepts—particularly the Wasserstein problem—and its applications in generative models.

## 2. Optimal Transport Problems

### 2.1 Monge's Problem

Consider excavating earth from a pit to form a mound. The goal is to minimize transportation cost (e.g., physical effort). Mathematically, given:
- A metric space $(X, d)$  
- Probability measures $\mu, \nu$ (source/target distributions, e.g., pit and mound)  
- Cost function $c: X \times X \to [0, \infty)$  

We seek a measure-preserving map $T: X \to X$ (where $T_\#\mu = \nu$) to minimize:
$$
\min_{T: T_\# \mu = \nu} \int_X c(x, T(x)) \, d\mu(x)
$$

**Limitations**:  
1. Non-convex optimization (hard to solve).  
2. No valid $T$ may exist for non-absolutely continuous measures.

### 2.2 Kantorovich's Relaxation

Kantorovich introduced **couplings** (joint distributions) to address Monge's limitations. For $\gamma \in X \times X$ with marginals $\mu, \nu$, the problem becomes:
$$
\min_{\gamma} \int_{X \times X} c(x, x') \, d\gamma(x, x') \quad \text{s.t.} \quad P_{x\#}\gamma = \mu, \ P_{x'\#}\gamma = \nu
$$

**Dual Formulation**:  
Find potentials $f, g$ satisfying $f(x) + g(x') \leq c(x, x')$, yielding:
$$
\max_{f, g} \left( \int_X f(x) \, d\mu(x) + \int_X g(x') \, d\nu(x') \right)
$$
This duality underpins the **Wasserstein-1 distance** (Earth Mover's Distance, EMD).

## 3. Wasserstein Distance

### 3.1 Definition

For cost $c(x, x') = d(x, x')^p$ ($p \geq 1$), the Wasserstein-$p$ distance is:
$$
W_p(\mu, \nu) = \left( \inf_{\gamma} \int_{X \times X} d(x, x')^p \, d\gamma(x, x') \right)^{1/p}
$$

### 3.2 Key Properties

1. **Metric axioms**: Non-negativity, symmetry, triangle inequality.  
2. **Weak convergence**: If $\mu_n \rightharpoonup \mu$, then $W_p(\mu_n, \mu) \to 0$.  
3. **Moment convergence**: $W_p$ convergence implies moment convergence.  

**Advantage over KL-divergence**:  
$W_p$ remains finite for non-overlapping supports, making it ideal for generative model losses.

### 3.3 Kantorovich-Rubinstein Theorem

The dual form of $W_1$ distance:
$$
W_1(\mu, \nu) = \sup_{f \in \text{Lip}_1(X)} \int_X f(x) \, d(\mu - \nu)(x)
$$
where $\text{Lip}_1(X)$ is the set of 1-Lipschitz functions. This form is computationally tractable for generative models.

## 4. Applications in Generative Models

### 4.1 Wasserstein GAN (WGAN)

WGAN uses the KR duality to design its loss function. The discriminator $D_\theta$ parameterizes the 1-Lipschitz function $f$, while the generator $G_\phi$ minimizes $W_1$:
$$
\min_G \max_{D \in \text{Lip}_1} \left( \mathbb{E}_{x \sim P_{\text{data}}} [D(x)] - \mathbb{E}_{z \sim p(z)} [D(G(z))] \right)
$$
**Advantage**: Superior to traditional GANs (using MSE loss) because it directly minimizes distributional divergence.

### 4.2 Rectified Flow

*To be expanded. For connections between Rectified Flow and OT, see:*  
[Rectified Flow: A Marginal Preserving Approach to Optimal Transport](https://arxiv.org/abs/2209.14577)