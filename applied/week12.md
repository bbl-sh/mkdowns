# Week 12

This document covers the fundamentals of accelerating deployments using TensorRT, including notes and code examples from a PDF.
Some maths are not working, i will correct this later on(i used chatgpt but it wrote in latex which is again not supported in markdown)

# Word Semantics and Embedding

## Computational Semantics
Automates meaning representation. Distributional semantics: word meaning from usage (Wittgenstein), synonymy from similar contexts (Harris, 1954). "Lemon is a rich source..." and "Bergamot is often used..." → bergamot as citrus. Dated 07/10/23.

## Distributional Representation
Words as vectors in semantic space. Co-occurrence matrices: term-document ("cherry": [2, 8, 9, 442, 25] vs. "digital": [1670, 1683, 85, 5, 4] for "computer," "data," "result," "pie," "sugar") or word-context. Sparse, high-dimensional (50k words).

## Word2Vec
Mikolov et al. (2013). Dense vectors via neural network. CBOW: predict target from context ("tablespoon of ___ jam" → "apricot"). Skip-gram: predict context from target ("apricot" → "tablespoon," "jam"). "...lemon, a [tablespoon of apricot jam, a] pinch..." → pairs (apricot, tablespoon). Self-supervised. Shifts: "dread" from awe to fear (William et al., 2016).

## Relational Properties and Bias
Analogies: "king - man + woman ≈ queen" (Word2Vec), cosine similarity:
\[
\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}
\]
Bias: GloVe links "Leroy" to "unpleasant," "Brad" to "pleasant" (ethnic harm). Word2Vec: "man:programmer :: woman:homemaker" (gender harm). Debiasing proposed (Bolukbasi et al., 2016).

## BERT
Devlin et al. (2019). Bidirectional transformers, 12 layers, 12 attention heads. MLM: 15% tokens masked (80% [MASK], 10% random, 10% unchanged), predict originals ("The [MASK] ran" → "dog"). NSP: classify sentence pairs. Contextual embeddings vs. static.

## Fine-Tuning
Add output layer, train on task data (e.g., classification). Movie reviews: "...zany characters..." → Positive, "pathetic..." → Negative. Input: document \(d\), classes \(C = \{c_1, c_2, ..., c_J\}\), training \((d_1, c_1), ..., (d_m, c_m))\).
