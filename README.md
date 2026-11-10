# Legal Language Modelling and NLP for Indian Legal Documents

**CS781 — Legal Language Modelling Capstone Project**

This repository contains the implementation and experiments for a multi-phase legal NLP project focused on building language models for Indian legal documents and adapting them to downstream legal reasoning and information-processing tasks.

The project follows a two-stage approach:

1. **Legal language model pretraining:** Develop and train language models from scratch using a corpus of Indian legal documents.
2. **Downstream task adaptation:** Fine-tune and adapt the pretrained models for seven legal NLP tasks, exploring full fine-tuning, parameter-efficient fine-tuning, and knowledge distillation.

The project investigates how different model architectures, document representations, context-handling strategies, and adaptation techniques affect performance across a range of legal NLP problems.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Phase 1: Legal Language Model Pretraining](#phase-1-legal-language-model-pretraining)
* [Model Architectures](#model-architectures)
* [Downstream Tasks](#downstream-tasks)

  * [Phase 2: Legal Named Entity Recognition](#phase-2-legal-named-entity-recognition)
  * [Phase 3: Rhetorical Role Prediction](#phase-3-rhetorical-role-prediction)
  * [Phase 4: Court Judgment Prediction](#phase-4-court-judgment-prediction)
  * [Phase 5: Legal Statute Identification](#phase-5-legal-statute-identification)
  * [Phase 6: Abstractive Legal Summarization](#phase-6-abstractive-legal-summarization)
  * [Phase 7: Prior Case Retrieval](#phase-7-prior-case-retrieval)
* [Fine-Tuning and Adaptation Strategies](#fine-tuning-and-adaptation-strategies)
* [Preprocessing and Long-Document Handling](#preprocessing-and-long-document-handling)
* [Evaluation](#evaluation)

---

## Project Overview

Legal documents present several challenges for natural language processing:

* They are often substantially longer than ordinary text sequences.
* Their terminology, structure, and writing style differ from general-purpose text.
* Understanding a legal document frequently requires information distributed across multiple paragraphs or sections.
* Many tasks require more than simple text classification: they involve extracting exact spans, identifying multiple applicable statutes, generating summaries, or ranking related cases.

The objective of this project is to investigate these challenges using language models pretrained specifically on Indian legal text.

Rather than relying on externally pretrained language models, the project develops its own language models and uses them as the starting point for all downstream tasks. This makes it possible to study how architectural choices made during pretraining influence performance on different legal NLP problems.

The project consists of seven phases:

| Phase | Task                             | Problem type               |
| ----- | -------------------------------- | -------------------------- |
| 1     | Legal Language Model Pretraining | Language modelling         |
| 2     | Legal Named Entity Recognition   | Span extraction            |
| 3     | Rhetorical Role Prediction       | Sentence classification    |
| 4     | Court Judgment Prediction        | Binary classification      |
| 5     | Legal Statute Identification     | Multi-label classification |
| 6     | Abstractive Legal Summarization  | Text generation            |
| 7     | Prior Case Retrieval             | Information retrieval      |

## Phase 1: Legal Language Model Pretraining

The first phase focuses on developing legal language models from scratch.

### Training corpus

The pretraining corpus consists of four sources of Indian legal text:

* Acts and statutory material
* High Court judgments
* Supreme Court judgments
* Tribunal documents

A custom byte-level Byte Pair Encoding (BPE) tokenizer is used to represent the legal text. The tokenizer preserves casing and punctuation while allowing frequently occurring character sequences and legal expressions to be represented efficiently.

The training pipeline includes:

* Corpus preprocessing and tokenization.
* Construction of training and validation sequences.
* Autoregressive next-token prediction.
* Model checkpointing and validation.
* Text generation and evaluation.

### Training objective

The language models are trained using the standard next-token prediction objective:

$$
\mathcal{L}_{\mathrm{LM}}
=
-\sum_{t=1}^{T}
\log P(x_t\mid x_1,\ldots,x_{t-1}).
$$

The objective is to learn representations of legal language that can subsequently be transferred to the downstream tasks.

## Model Architectures

Four substantially different architectures were developed and pretrained:

| Architecture                       | Description                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| Decoder-only Transformer           | Uses causal self-attention to predict the next token from preceding tokens.               |
| Encoder–decoder Transformer        | Uses an encoder to represent input sequences and a decoder to generate output sequences.  |
| Hybrid GRU–Attention model         | Combines recurrent sequence modelling with attention-based contextual representations.    |
| Lightweight parameter-shared model | Uses parameter sharing and sliding-window attention to reduce computational requirements. |

The hybrid architecture uses recurrent and attention-based components to combine sequential modelling with contextual information. One configuration explored a hidden dimension of 768, a vocabulary of 30,000 tokens, and a context length of 1,024 tokens.

All four models are trained from randomly initialized weights rather than relying on externally pretrained language-model checkpoints.

The resulting checkpoints and tokenizer are used as the foundation for subsequent phases.

---

## Downstream Tasks

All downstream experiments begin with the language models developed in Phase 1. Task-specific prediction heads, adaptation modules, or retrieval objectives are added depending on the problem.

### Phase 2: Legal Named Entity Recognition

**Task:** Identify legally relevant entities in Indian legal documents and predict their exact character-level spans and entity types.

Unlike conventional named entity recognition, legal NER focuses on entities that play specific roles in legal proceedings, including parties, judges, counsels, courts, statutes, precedents, witnesses, dates, and case identifiers.

The model predicts:

* Entity start and end character offsets.
* The corresponding entity label.

**Approach:**

* Token-level sequence labelling using BIO-style tags.
* Alignment between tokenizer offsets and character-level annotations.
* Sliding-window processing for long documents.
* Overlap handling when combining predictions from multiple windows.
* Span decoding and conversion back to the original document's character offsets.

Exact span matching is important: a prediction with an incorrect character boundary can be counted as incorrect even if the entity is semantically identified correctly.

**Evaluation:** Strict span-level F1, with precision and recall also considered.

### Phase 3: Rhetorical Role Prediction

**Task:** Identify the rhetorical function of every sentence in a legal document.

A legal judgment contains sentences serving different purposes, such as presenting facts, raising issues, describing arguments, referring to legal provisions, or providing judicial reasoning and conclusions.

The task covers two legal domains:

* Corporate Law (CL)
* Information Technology (IT)

**Approach:**

* Sentence-level classification using pretrained legal representations.
* Experiments with sentence-only and context-aware representations.
* Sentence packing to process multiple sentences efficiently.
* Contextual modelling using surrounding sentences.
* Task-specific classification heads and adaptation strategies.

The central challenge is distinguishing between rhetorically similar sentences while retaining enough document context to understand their roles.

**Evaluation:** Macro-F1.

### Phase 4: Court Judgment Prediction

**Task:** Predict the outcome of a court case from its legal document.

This is formulated as a binary classification problem, with the model predicting one of two possible judgment labels.

**Approach:**

* Adaptation of pretrained legal language models.
* Document-level classification heads.
* Chunk-based processing for long judgments.
* Experiments with different chunk representations and pooling strategies.
* Comparison of full fine-tuning and parameter-efficient adaptation.

Long judgments may contain relevant evidence in multiple sections. Consequently, an important part of this phase is investigating how information from different parts of a document can be combined into a useful representation for classification.

**Evaluation:** Binary classification metrics prescribed by the competition.

### Phase 5: Legal Statute Identification

**Task:** Identify all relevant statute IDs associated with a legal document.

Unlike ordinary single-label classification, a document can be associated with multiple statutes. The task is therefore formulated as multi-label classification.

**Approach:**

* A multi-label prediction head with independent statute scores.
* Sigmoid-based probability prediction.
* Binary cross-entropy-based training.
* Experiments with class weighting and threshold selection.
* Chunk-based document representations for long judgments.
* Different pooling strategies for aggregating information across chunks.

The model must identify multiple applicable statutes rather than selecting only the most likely one. Particular attention is therefore required for class imbalance and infrequently occurring statutes.

**Evaluation:** Multi-label classification metrics prescribed by the competition.

### Phase 6: Abstractive Legal Summarization

**Task:** Generate a concise summary of a legal document while preserving its important facts, arguments, and reasoning.

This is a sequence-generation task in which the model receives a legal document and generates its corresponding summary.

**Approach:**

* Adaptation of pretrained language models for conditional text generation.
* Document-to-summary training.
* Summary-only loss computation, so that the model is trained to generate the target summary rather than reproduce the input document.
* Experiments with different document truncation and selection strategies.
* Head–tail selection and importance-based context selection.
* Hierarchical chunk-based summarization.
* Generation controls to manage output length and repetition.

A major challenge is handling documents that exceed the model's context length without losing important legal information.

**Evaluation:**

* BERTScore F1 (primary metric).
* ROUGE-1 F1.
* ROUGE-L F1.

### Phase 7: Prior Case Retrieval

**Task:** Retrieve and rank legally relevant prior cases for a given query case.

The model receives a query case and a candidate pool of prior cases. It must produce a ranked list of candidate case IDs based on legal relevance.

**Approach:**

* A dual-encoder architecture with shared model weights.
* Separate representations for query and candidate documents.
* Projection into a shared embedding space.
* L2-normalized document embeddings.
* Contrastive learning using an InfoNCE-style objective.
* In-batch negatives and hard-negative sampling.
* Similarity-based ranking using query–candidate embeddings.

Long documents are represented using chunk-based processing and aggregation. Experiments also investigate combining document-level similarity with chunk-level similarity to capture relevant information that may otherwise be diluted in a single document embedding.

The retrieval system restricts predictions to the candidate pool supplied for each query.

**Evaluation:** Ranking and retrieval metrics prescribed by the competition.

---

## Fine-Tuning and Adaptation Strategies

A major component of the project is comparing different ways of adapting the pretrained language models to downstream tasks.

The experiments investigate both conventional fine-tuning and parameter-efficient approaches.

| Method                 | Description                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| Full fine-tuning       | Updates all model parameters using the downstream task.                                   |
| Partial fine-tuning    | Updates selected layers while keeping the remaining parameters frozen.                    |
| LoRA                   | Introduces trainable low-rank updates to selected weight matrices.                        |
| LoRMA+                 | Explores the LoRMA-based adaptation approach used in the project.                         |
| Adapters               | Adds small trainable modules within the existing model architecture.                      |
| Prompt tuning          | Learns trainable prompt representations while keeping the base model largely frozen.      |

LoRA represents weight updates as a low-rank decomposition:

$$
\Delta W = BA,
$$

where the trainable matrices \(A\) and \(B\) have a smaller intermediate rank than the original weight matrix.

These approaches are compared in terms of task performance, trainable parameter count, memory requirements, and computational efficiency.

Knowledge distillation is investigated as a way of transferring useful task-specific behaviour while reducing the size of the resulting model.

---

## Preprocessing and Long-Document Handling

Legal documents frequently exceed the context length supported by the language models. The project therefore explores multiple strategies for representing long documents.

### Tokenization

A custom byte-level BPE tokenizer is used throughout the project. It provides a shared vocabulary for the pretrained models and downstream tasks.

### Sliding windows

Long documents are divided into overlapping token windows. Overlap helps preserve contextual information near window boundaries and reduces the risk of losing entities or other relevant content split across chunks.

### Chunk aggregation

For document-level tasks, representations from multiple chunks are combined to obtain a single document representation. Different pooling approaches are investigated, including mean pooling and learned aggregation.

### Sentence-level context

For rhetorical role prediction, sentence boundaries are retained and surrounding sentences can be incorporated as context. This allows the model to use information beyond the sentence being classified.

### Selective context

For generation and other long-document tasks, the project also explores selecting important portions of a document instead of simply truncating everything after the context limit.

These strategies are particularly relevant to judgment prediction, statute identification, summarization, and prior case retrieval.

---

## Evaluation

Each phase is evaluated using metrics appropriate to its task.

| Phase | Task                       | Evaluation                                          |
| ----- | -------------------------- | --------------------------------------------------- |
| 1     | Language modelling         | BertScore                                           |
| 2     | Legal NER                  | Strict Macro-F1                                     |
| 3     | Rhetorical role prediction | Macro-F1                                            |
| 4     | Judgment prediction        | Macro-F1                                            |
| 5     | Statute identification     | Macro-F1                                            |
| 6     | Summarization              | BERTScore F1                                        |
| 7     | Prior case retrieval       | Micro-F1@7                                          |

Local validation is used for checkpoint selection and experimental comparisons. Final submissions follow the corresponding Codabench submission formats.

---

## Acknowledgements

This project was developed as part of **CS781: Legal Language Modelling**, under the guidance of Prof. Ashutosh Modi, at the Indian Institute of Technology Kanpur.

The project uses the datasets and task specifications provided through the course competition.
