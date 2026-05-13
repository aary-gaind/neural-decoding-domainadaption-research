# Multimodal EEG Representation Learning for Cognitive Workload  
### Cross-Subject Generalization, Few-Shot Adaptation, Cross-Dataset Alignment, and Wearable EEG Distillation

---

## 1. Project Overview

This project focuses on building EEG models that generalize across people, sessions, datasets, and hardware setups for cognitive workload decoding.

The central problem is that EEG models often perform well within one subject or one recording setup, but fail when tested on a new person. This is a major barrier for real-world neurotechnology and wearable brain-computer interfaces.

This work attacks that problem through:

- Multimodal EEG feature learning
- Cross-subject supervised contrastive learning
- Prototype-based classification
- Ordinal workload structure
- Leave-one-subject-out evaluation
- Few-shot adaptation with AdaBN and Mahalanobis prototypes
- Cross-dataset alignment
- Teacher-student distillation from full EEG to low-channel wearable EEG

The long-term goal is to move from full-scalp EEG models toward practical wearable EEG systems that can decode cognitive effort and workload in real-world settings.

---

## 2. Core Motivation

EEG is noisy, non-stationary, and highly subject-dependent. Two people performing the same workload task can produce very different raw signal distributions because of anatomy, electrode contact, neural variability, session drift, and hardware noise.

Because of this, standard EEG classifiers often learn shortcuts such as:

- Subject identity
- Session-specific signal statistics
- Dataset-specific artifacts
- Hardware-specific noise patterns

Instead of only training a classifier, this project treats EEG decoding as a representation learning problem.

The goal is to learn an embedding space where cognitive workload states form stable, structured clusters that transfer across subjects and datasets.

---

## 3. Core Hypothesis

Cognitive workload has an underlying geometric structure that can be learned from EEG.

Rather than treating Low, Medium, and High workload as unrelated classes, the model should learn that:

- Low and Medium are closer than Low and High
- Medium lies between Low and High
- Workload has an ordinal structure
- Subject-specific signal variation should be separated from task-relevant cognitive variation

The model is therefore trained to learn embeddings that are:

- Class-discriminative
- Subject-invariant
- Ordinally structured
- Robust to distribution shift
- Adaptable with only a small number of samples from a new user

---

## 4. Dataset Setup

The project uses multiple EEG workload datasets to test whether the learned representations generalize beyond a single controlled setting.

### 4.1 N-Back Workload Dataset

The n-back dataset is used for working-memory-based cognitive workload classification.

It contains EEG recordings across multiple workload levels, where higher n-back difficulty corresponds to higher cognitive demand.

This dataset is useful because it has a clear workload structure and strong temporal dynamics.

### 4.2 Arithmetic EEG Dataset

A second dataset is based on arithmetic workload conditions.

The arithmetic data uses 19 EEG channels sampled at 250 Hz, with channels including:

Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2

The original labels are binned into three workload classes:

- Low
- Medium
- High

This dataset is especially useful for testing cross-subject generalization because arithmetic difficulty creates graded cognitive demand.

### 4.3 Mendeley Arithmetic and Stroop Dataset

This dataset contains EEG recordings from arithmetic and Stroop tasks using an OpenBCI Cyton setup.

It includes 8 channels:

Fp1, Fp2, F7, F3, FZ, F4, F8, C2

Each subject performs different task difficulty levels:

- Natural
- Low
- Medium
- High

This dataset is valuable because it is closer to a lower-channel, practical EEG setup and introduces another domain shift compared to the n-back and 19-channel arithmetic datasets.

---

## 5. Preprocessing and Quality Control

A major part of this project involved building a reliable EEG preprocessing and QC pipeline before modeling.

The preprocessing pipeline includes:

- 60 Hz notch filtering
- 4–35 Hz bandpass filtering
- Windowing into fixed-length EEG segments
- Train-only standardization
- Artifact and spike detection
- Per-window rejection analysis
- Visualization of kept vs. rejected windows

The project also explored referencing and artifact-removal strategies, including:

- Common average reference
- ICA-based artifact removal
- Reconstruction error checks
- Bandpower before/after comparisons
- Rank-drop analysis after rereferencing

This was important because EEG model performance depends heavily on whether the model is learning brain-related signal or artifact-driven shortcuts.

---

## 6. Multimodal Feature Design

Each EEG window is represented using multiple complementary views of the signal.

### 6.1 Time-Domain EEG

The raw EEG window captures temporal neural dynamics directly.

This branch allows the model to learn waveform-level patterns, local oscillations, transient changes, and temporal dependencies.

### 6.2 Bandpower Features

Bandpower features summarize energy in canonical EEG frequency bands.

These include:

- Delta
- Theta
- Alpha
- Beta
- Gamma

Bandpower is important because workload is often associated with changes such as increased frontal theta and reduced alpha power.

### 6.3 Differential Entropy Features

Differential entropy features provide another frequency-domain representation of EEG complexity and band-specific activity.

These are useful because DE features are commonly used in EEG affective and workload decoding.

### 6.4 Hjorth Features

Hjorth parameters capture signal shape and dynamics through:

- Activity
- Mobility
- Complexity

These features describe the variance, frequency content, and irregularity of the EEG signal.

### 6.5 Spectral and Statistical Features

Additional handcrafted features include:

- Spectral entropy
- Spectral centroid
- Mean absolute value
- Zero-crossing rate
- Band ratios
- Frontal theta / parietal alpha summaries
- Alpha asymmetry-style features

### 6.6 Covariance Features

Covariance features capture spatial relationships between EEG channels.

They help model how activity across electrodes changes together under different workload states.

### 6.7 Functional Connectivity Features

Phase Locking Value features capture synchronization between channels.

This gives the model access to functional connectivity information, not just local electrode activity.

---

## 7. Model Architecture

The model uses a multimodal fusion architecture.

Each modality is processed by a separate encoder branch, and the resulting embeddings are fused into one shared representation.

### 7.1 Time Encoder

The time-domain branch uses convolutional neural networks designed for EEG.

It includes:

- Depthwise temporal convolutions
- Multi-scale temporal kernels
- Residual temporal blocks
- Channel attention
- Temporal attention pooling

The purpose of this branch is to learn both short-timescale and long-timescale EEG patterns.

### 7.2 Feature Encoder

The handcrafted feature branch uses MLP layers with normalization and dropout.

It processes bandpower, differential entropy, Hjorth features, spectral features, and other summary statistics.

This branch gives the model stable EEG descriptors that are often more robust than raw time-domain learning alone.

### 7.3 Covariance and Connectivity Encoders

Covariance and PLV features are processed through separate MLP encoders.

These branches help the model understand spatial and connectivity-level structure across the EEG montage.

### 7.4 Fusion Layer

The outputs from all branches are concatenated and projected into a compact embedding space.

The final embedding is used for:

- Prototype classification
- Contrastive learning
- Few-shot adaptation
- Mahalanobis classification
- Teacher-student distillation

---

## 8. Representation Learning Objective

The training objective combines several losses to shape the EEG embedding space.

### 8.1 Prototype Classification

Each workload class is represented by a prototype, computed as the centroid of embeddings from that class.

Classification is performed by comparing each sample embedding to the class prototypes.

This encourages each workload state to occupy a stable region of the embedding space.

### 8.2 Cross-Subject Supervised Contrastive Learning

The contrastive loss treats samples as positives when they share the same workload label but come from different subjects.

This is important because it directly discourages the model from relying on subject-specific shortcuts.

The model is pushed to learn what Low, Medium, and High workload look like across people, not just within one person.

### 8.3 Prototype Compactness

A compactness loss pulls embeddings toward their class prototype.

This improves cluster tightness and makes few-shot adaptation more stable.

### 8.4 Ordinal Workload Structure

Because workload is ordered, the model can use ordinal structure instead of treating the classes as completely independent.

The project explores ordinal losses and distance-aware contrastive learning so that:

- Low and Medium are allowed to be closer
- Medium and High are allowed to be closer
- Low and High are pushed farther apart

This better matches the real structure of cognitive workload.

### 8.5 Combined Objective

The full training objective combines:

- Prototype cross-entropy
- Supervised contrastive loss
- Prototype compactness
- Ordinal structure regularization
- Distance-aware contrastive learning

This produces a more structured and generalizable EEG embedding space.

---

## 9. Cross-Dataset Alignment

A key extension of the project is aligning workload geometry across datasets.

Different datasets may have different signal statistics, hardware setups, channel layouts, task structures, and label distributions.

To handle this, the model encourages consistency between class-prototype relationships across datasets.

The goal is not just to classify one dataset well, but to learn a workload representation that remains meaningful across different experimental conditions.

---

## 10. Evaluation Framework

The main evaluation strategy is leave-one-subject-out cross-validation.

For each fold:

1. One subject is held out completely.
2. The model trains on all other subjects.
3. The held-out subject is used for testing.
4. Zero-shot accuracy is measured first.
5. Few-shot adaptation is then performed using a small support set from the held-out subject.

This evaluation is much stricter than random train/test splitting because the model must generalize to a person it has never seen before.

---

## 11. Zero-Shot Evaluation

Zero-shot evaluation tests whether the model can classify a new subject without any adaptation.

This measures pure cross-subject generalization.

The teacher model achieved meaningful zero-shot performance across held-out subjects, showing that the learned embedding space captures transferable workload structure.

However, zero-shot accuracy still varies across subjects because EEG subject shift remains a difficult problem.

---

## 12. Few-Shot Adaptation

Few-shot adaptation uses a small labeled support set from the held-out subject.

The adaptation procedure is:

1. Run the held-out subject support samples through the encoder.
2. Compute class prototypes from the support embeddings.
3. Estimate covariance in the embedding space.
4. Apply shrinkage for numerical stability.
5. Classify query samples using Mahalanobis distance.

This allows the model to adapt to a new user without full retraining.

### 12.1 Adaptive BatchNorm

Adaptive BatchNorm updates batch normalization statistics using the target subject’s data.

No model weights are updated.

This helps correct distribution shift between training subjects and the held-out test subject.

### 12.2 Mahalanobis Prototype Classification

Mahalanobis distance improves over simple Euclidean distance because it accounts for feature covariance.

This is useful when embedding dimensions are correlated or have different noise levels.

### 12.3 Covariance Shrinkage

Shrinkage stabilizes covariance estimation in few-shot settings.

This prevents the covariance matrix from becoming unstable when only a small number of support samples are available.

---

## 13. Teacher Model Results

The full teacher model uses the richer EEG setup and multimodal feature representation.

Across 13 leave-one-subject-out folds, the teacher model achieved:

- Mean AdaBN + Mahalanobis few-shot accuracy: 75.68%
- Standard deviation: 8.18%

Per-subject few-shot accuracies:

- Subject 01: 78.45%
- Subject 02: 84.76%
- Subject 03: 76.98%
- Subject 04: 78.88%
- Subject 06: 78.48%
- Subject 07: 73.17%
- Subject 08: 82.64%
- Subject 09: 60.49%
- Subject 10: 92.05%
- Subject 12: 71.06%
- Subject 13: 70.45%
- Subject 14: 63.41%
- Subject 15: 73.04%

These results show that the combination of representation learning, AdaBN, and Mahalanobis few-shot adaptation substantially improves held-out-subject performance.

---

## 14. Teacher-Student Distillation

The full teacher model is powerful but not ideal for wearable deployment because it uses more EEG channels and richer features.

The student model is designed to use fewer channels while preserving as much of the teacher’s representation as possible.

### 14.1 Student Motivation

A wearable EEG system cannot rely on a full research-grade EEG montage.

The student model explores whether a compact channel subset can approximate the teacher model’s behavior.

This is a step toward:

- Ear-EEG
- Around-ear EEG
- Frontal wearable EEG
- Low-channel consumer neurotechnology

### 14.2 Student Inputs

The student model uses a reduced EEG channel set, such as frontal or wearable-relevant channels.

The goal is to preserve cognitive workload decoding performance while reducing hardware complexity.

### 14.3 Distillation Losses

The student is trained using:

- Standard supervised classification loss
- Prototype loss
- Contrastive loss
- Logit distillation from the teacher
- Feature alignment with teacher embeddings

Logit distillation transfers the teacher’s decision boundaries.

Feature alignment transfers the teacher’s embedding geometry.

Together, these help the student model behave like the full teacher model even with fewer channels.

---

## 15. Why This Matters for Wearable Neurotechnology

This project is not only about improving benchmark accuracy.

It is about building the technical foundation for a real wearable EEG system that can estimate cognitive effort, workload, and mental state in everyday environments.

The key idea is:

Train a powerful full-EEG teacher model first, then compress its knowledge into a smaller wearable student model.

This creates a path from research-grade EEG to practical consumer neurotechnology.

---

## 16. Key Technical Contributions

This project includes the following technical components:

1. Built a multimodal EEG representation learning pipeline combining raw time signals, bandpower, differential entropy, Hjorth features, covariance, and connectivity.

2. Implemented cross-subject supervised contrastive learning to reduce subject-specific overfitting.

3. Used prototype-based classification to structure the embedding space around workload class centroids.

4. Added ordinal workload modeling so Low, Medium, and High workload are treated as ordered cognitive states.

5. Evaluated rigorously using leave-one-subject-out cross-validation.

6. Added AdaBN and Mahalanobis few-shot adaptation for new-subject calibration.

7. Built preprocessing and QC tools for filtering, artifact detection, ICA inspection, and rejected-window visualization.

8. Explored cross-dataset alignment across n-back, arithmetic, and Stroop/arithmetic workload datasets.

9. Developed a teacher-student distillation framework for transferring full-EEG knowledge into low-channel wearable EEG models.

---

## 17. Main Insight

The biggest insight from this work is that EEG workload decoding should not be treated as simple classification.

A better framing is:

Learn a stable cognitive-state geometry that survives subject shift, dataset shift, and hardware reduction.

This means the important object is not just the classifier head.

The important object is the embedding space.

If the embedding space is structured well, then adaptation becomes easier, few-shot learning becomes more reliable, and distillation into wearable systems becomes possible.

---

## 18. Future Work

Future directions include:

- Scalp EEG to ear-EEG distillation
- Testing around-ear channel layouts
- Adding eye tracking and pupil dilation
- Adding HRV or ECG features
- Real-time streaming inference
- Online adaptation for session drift
- Riemannian alignment and covariance-based layers
- More robust cross-dataset workload alignment
- Personalized calibration with very small support sets
- Deployment-oriented lightweight models

---

## 19. Final Takeaway

This project builds a full EEG workload decoding pipeline designed for real generalization.

It moves from raw EEG classification toward structured representation learning, few-shot personalization, and wearable EEG distillation.

The long-term vision is:

Train on rich EEG, learn stable cognitive-state geometry, adapt quickly to new users, and distill into low-channel wearable neurotechnology.
