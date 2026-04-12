# 🧠 Multimodal EEG Representation Learning for Cognitive Workload  
### Cross-Subject Generalization, Cross-Dataset Alignment, and Teacher–Student Distillation

---

## 1. Introduction

Electroencephalography (EEG) offers a non-invasive window into brain activity, enabling the decoding of cognitive states such as workload, attention, and fatigue. However, despite decades of research, EEG-based systems still struggle with a fundamental limitation:

> **Poor generalization across subjects, sessions, and recording setups.**

This limitation arises from:
- High inter-subject variability  
- Non-stationarity of neural signals  
- Sensitivity to hardware, placement, and noise  
- Weak signal-to-noise ratio  

As a result, most EEG models:
- Overfit to individuals  
- Fail in real-world deployment  
- Cannot scale to consumer or wearable systems  

---

## 2. Core Hypothesis

This project is built on the following hypothesis:

> **Cognitive states (e.g., workload) have an underlying geometric structure that can be learned and shared across subjects and datasets.**

Rather than treating EEG decoding as pure classification, we instead:

- Learn a **structured embedding space**
- Enforce **cross-subject alignment**
- Preserve **inter-class geometry**
- Enable **few-shot adaptation**
- Transfer knowledge to **low-channel systems**

---

## 3. System Overview

The pipeline consists of four major stages:

---

### Stage 1 — Multimodal Representation Learning
Learn embeddings from full-scalp EEG using:
- Time-domain signals  
- Frequency-domain features  
- Spatial covariance  
- Functional connectivity (PLV)  

---

### Stage 2 — Metric + Contrastive Learning
Train embeddings using:
- Prototype-based classification  
- Cross-subject contrastive learning  
- Prototype compactness  

---

### Stage 3 — Adaptation at Inference
Handle distribution shift via:
- Embedding shift detection  
- Adaptive BatchNorm (AdaBN)  
- Few-shot prototype estimation  
- Mahalanobis distance classification  

---

### Stage 4 — Teacher–Student Distillation
Transfer knowledge from:
- Full 14-channel EEG → 3-channel EEG  

Enabling:
> **Towards wearable EEG systems**

---

## 4. Dataset Setup

The system is evaluated using two distinct cognitive workload datasets:

---

### 4.1 N-Back Dataset
- Standard working memory paradigm  
- Multiple workload levels  
- Strong temporal structure  

---

### 4.2 HTC Dataset
- More complex and heterogeneous workload  
- Different experimental conditions  
- Strong domain shift relative to N-back  

---

### 4.3 Cross-Dataset Challenge

The datasets differ in:
- Task structure  
- Signal statistics  
- Label distributions  

This creates a **domain generalization problem**:
> A robust model must learn representations that are invariant across datasets.

---

## 5. Multimodal Feature Design

Each EEG sample is represented using four complementary modalities:

---

### 5.1 Time-Domain EEG
Shape: (14, 512)

Captures:
- Raw neural oscillations  
- Temporal dependencies  
- Local transient patterns  

---

### 5.2 Bandpower Features
Shape: (42,)

Captures:
- Frequency energy distribution  
- Cognitive state correlates (e.g., theta ↑ with workload)  

---

### 5.3 Covariance Features
Shape: (105,)

Captures:
- Channel-to-channel correlations  
- Global spatial structure  

---

### 5.4 Phase Locking Value (PLV)
Shape: (210,)

Captures:
- Phase synchronization  
- Functional connectivity  

---

### Why This Matters

EEG signals are inherently multi-dimensional:

| Dimension | Example |
|----------|--------|
| Time | Waveforms |
| Frequency | Bandpower |
| Space | Covariance |
| Connectivity | PLV |

> A single representation is insufficient — multimodal fusion is essential.

---

## 6. Model Architecture

---

### 6.1 Fusion Encoder

The encoder processes each modality independently before fusion.

---

#### Time Branch

- Depthwise convolutions (per-channel filtering)
- Temporal kernels (short + long range)
- Channel attention mechanism
- Learned temporal attention pooling

Key idea:
> Learn both **local patterns** and **global temporal importance**

---

#### Bandpower Branch

- Fully connected layers
- BatchNorm + dropout

Encodes:
> Stable frequency-domain structure

---

#### Covariance Branch

- MLP over covariance vector

Encodes:
> Spatial relationships across electrodes

---

#### PLV Branch

- MLP over connectivity features

Encodes:
> Functional connectivity patterns

---

### 6.2 Fusion

All modality embeddings are concatenated:
z = [z_time, z_bp, z_cov, z_plv]

Then projected:

e ∈ ℝ⁶⁴

This embedding is:
- Compact  
- Multimodal  
- Trainable via metric learning  

---

### 6.3 Projection Head

Used only during training for contrastive learning:

- Encourages stable representation learning  
- Prevents collapse of embedding space  

---

## 7. Learning Framework

---

### 7.1 Prototype-Based Classification

Each class is represented by its centroid:
μ_c = mean(e_i | y_i = c)

Prediction is based on:
- Cosine similarity to prototypes  

---

### 7.2 Supervised Contrastive Learning

Positive pairs:
- Same class  
- **Different subjects**

This enforces:

> Subject-invariant representations of cognitive states

---

### 7.3 Prototype Compactness

Minimize:
|| e_i - μ_c ||²

Encourages:
- Tight clusters  
- Stable decision boundaries  

---

### 7.4 Combined Objective
L = L_proto_ce

*λ * L_contrastive
*α * L_compactness

---

## 8. Cross-Dataset Alignment

---

### Problem

Datasets may learn **different geometries** of workload.

---

### Solution: Distance Consistency

We align **pairwise distances between class prototypes** across datasets.

This ensures:
- Similar structure of cognitive states  
- Shared embedding geometry  

---

## 9. Training Strategy

---

### Balanced Batch Sampling

Each batch:
- Contains multiple subjects  
- Contains samples from both datasets  

Ensures:
- Stable contrastive learning  
- Cross-subject mixing  

---

### Subject Separation

Contrastive positives require:
- Same label  
- **Different subject**

Prevents:
> Shortcut learning via subject identity

---

## 10. Evaluation Framework

---

### 10.1 LOSO (Leave-One-Subject-Out)

For each subject:
- Train on all others  
- Test on held-out subject  

This is:
> The gold standard for EEG generalization

---

### 10.2 Zero-Shot Evaluation

- No adaptation  
- Direct prototype classification  

Measures:
> Pure generalization ability  

---

## 11. Distribution Shift Handling

---

### 11.1 Shift Detection
shift = || mean(train) - mean(test) ||

If shift is high:
→ domain mismatch detected  

---

### 11.2 Adaptive BatchNorm (AdaBN)

If shift exceeds threshold:
- Update BN statistics using test data  
- No weight updates  

Result:
> Fast, lightweight adaptation  

---

## 12. Few-Shot Adaptation

---

### Procedure

1. Sample small support set (e.g., 20 per class)
2. Compute prototypes
3. Estimate covariance
4. Classify queries

---

### 12.1 Mahalanobis Distance
d(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)

Advantages:
- Accounts for feature correlations  
- More robust than Euclidean  

---

### 12.2 Covariance Shrinkage
Σ' = (1 - α) Σ + α I

Improves:
- Numerical stability  
- Generalization  

---

### 12.3 Feature Reweighting

Dimensions weighted by inverse variance:
w_i = 1 / Var(e_i)

Effect:
> Downweights noisy features  

---

## 13. Teacher–Student Distillation

---

### 13.1 Motivation

Full model:
- High accuracy  
- Uses 14 channels  

But:
> Not deployable in wearable systems  

---

### 13.2 Student Model

Uses only:
[AF3, F3, FC5]

Architecture:
- Lightweight CNN  
- Bandpower branch  
- Smaller fusion network  

---

### 13.3 Distillation Objectives

---

#### Base Loss
Same as teacher:
- Prototype CE  
- Contrastive  
- Compactness  

---

#### Logit Distillation
KL(student || teacher)

Aligns decision boundaries  

---

#### Feature Alignment
1 - cosine_similarity(e_s, e_t)

Aligns embedding geometry  

---

### 13.4 Final Loss
L = L_base

0.5 * L_KL
0.25 * L_feature


---

## 14. Results

---

### Teacher Model

- Strong zero-shot generalization  
- Robust embeddings across subjects  
- Significant improvement with few-shot  

---

### Student Model

- Successfully mimics teacher behavior  
- Maintains strong performance with 3 channels  
- Demonstrates feasibility of low-channel decoding  

---

## 15. Key Insights

---

### 15.1 Geometry > Classification

EEG decoding works best when:
> The model learns a structured embedding space  

---

### 15.2 Cross-Subject Contrastive Learning is Essential

Without it:
- Model overfits to individuals  

---

### 15.3 Multimodal Fusion is Critical

Different features capture different aspects of brain activity  

---

### 15.4 Few-Shot + Mahalanobis is Extremely Effective

Avoids:
- Retraining  
- Overfitting  

---

### 15.5 Distillation Enables Wearable EEG

Train heavy → deploy light  

---

## 16. Future Work

---

### Ear-EEG Transfer
- Distill scalp EEG → ear EEG  

---

### Multimodal Expansion
- Eye tracking (pupil dilation)  
- HRV (ECG)  

---

### Advanced Geometry
- Riemannian methods  
- Manifold learning  

---

### Real-Time Systems
- Streaming inference  
- Low-latency pipelines  

---

## 17. Conclusion

This project presents a full pipeline for:

> **Learning, adapting, and compressing EEG representations for cognitive workload decoding**

It demonstrates:

- Cross-subject generalization  
- Cross-dataset robustness  
- Few-shot adaptability  
- Hardware-aware deployment  

---

## Final Takeaway

> Train powerful models on rich EEG → distill into simple, wearable systems  
