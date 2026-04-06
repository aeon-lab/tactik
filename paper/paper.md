---
title: 'TACTIK: Text Analysis Clustering Tuning Information and Keyword Extraction — An Optimized Pipeline for Text Clustering'
tags:
  - Python
  - text clustering
  - topic modeling
  - natural language processing
  - hyperparameter tuning
  - UMAP
  - HDBSCAN
authors:
  - name: Niklas P. Schulmeyer
    orcid: 0009-0000-0275-4728
    affiliation: 1
  - name: Nicoletta Fala
    orcid: 0000-0002-7392-4904
    affiliation: 1
affiliations:
  - index: 1
    name: Auburn University, United States of America
    ror: 01v91bn63
date: 4 April 2026
bibliography: paper.bib
---

# Summary

TACTIK is an open-source Python package providing an end-to-end pipeline for automated text clustering and topic analysis. The package combines efficient preprocessing, dimensionality reduction, and unsupervised clustering with keyword extraction and topic modeling. TACTIK integrates scalable hyperparameter tuning and memory optimization to handle large datasets and domain-specific applications. By modularizing each component, TACTIK supports both full-pipeline automation and standalone use of its individual modules. The software aims to accelerate text-based insight generation in research and applied analytics.

# Statement of Need

Text data is the basis for insights in many different fields, like sentiment analysis in markets [@RN1], pattern and trend analysis in incident reporting [@RN3; @RN2], and grouping and extracting common topics in news items [@RN4]. Textual datasets can vary considerably in structure depending on domain, submitting entity, or time of submission. For example, in aviation incident and accident reports, wording style and use of abbreviations can differ substantially [@RN5]. Dataset sizes also present challenges; the Aviation Safety Reporting System (ASRS) alone contains more than 1.9 million reports [@RN6].

A well-established approach for text clustering is the combination of UMAP for dimensionality reduction and HDBSCAN for clustering [@RN8; @RN7; @RN2]. Peer-reviewed research has evaluated this pipeline architecture and documented key configuration elements, such as reducing to no fewer than 15 dimensions [@RN8]. Some pipelines also integrate visualization, keyword extraction, and topic modeling to complement the clustering process [@RN7; @RN2].

Despite widespread adoption, three persistent challenges motivated the development of TACTIK: hyperparameter sensitivity [@RN9; @RN8; @RN10], the manual labeling bottleneck [@RN12; @RN7; @RN11], and scalability and topic separation difficulties [@RN15; @RN14; @RN13].

The UMAP-HDBSCAN pipeline, while effective, requires careful tuning of multiple parameters across both algorithms [@RN9; @RN8; @RN10]. Some researchers even argue that algorithms should be evaluated by the ease of tuning [@RN16]. Manual cluster labeling is time-consuming and costly, yet often necessary because automated methods fail to capture inherent structure [@RN12; @RN7; @RN11]. Lastly, pipelines frequently struggle to scale and to clearly separate topics when applied to large, heterogeneous datasets [@RN15; @RN14; @RN13]. TACTIK emerged from the need to create coherent, representative clusters from data such as the ASRS and to reliably extract information from them. It represents continued development of an earlier internal lab tool, enhanced through hyperparameter tuning and a three-fold topic extraction approach.

# State of the Field

The dominant tools for text clustering and topic modeling are LDA, NMF, Top2Vec, and BERTopic. Comparative studies have evaluated these approaches across domains, generally finding that embedding-based methods outperform classical bag-of-words models on coherence and interpretability [@RN17; @RN18]. However, a key limitation shared across these tools is the absence of systematic parameter optimization, with most implementations relying on default settings and repeated runs rather than principled tuning strategies.

TACTIK addresses this directly by integrating metric-driven hyperparameter tuning via the Davies-Bouldin and Calinski-Harabasz indices, giving researchers an interpretable and automated pathway to optimize cluster quality without requiring deep NLP expertise. This makes the pipeline accessible across domains well beyond aviation, including public health, legal analysis, and social science. Combined with three-fold topic extraction, TACTIK produces more interpretable outputs than the single-method representations used by BERTopic or Top2Vec, with less manual effort required from the user.

# Software Design

TACTIK prioritizes flexibility and research utility. The library is organized around six core pillars: preprocessing, clustering, tuning, topic modeling and labeling, visualization, and orchestration. Each pillar is encapsulated in a dedicated module that can be used as a standalone component, with the orchestration layer enabling full-pipeline use without requiring direct interaction with individual components. This design serves both expert users who need fine-grained control and novice users who prefer a single high-level call.

Preprocessing uses spaCy for tokenization, lemmatization, and part-of-speech tagging, and scikit-learn for vectorization [@RN20; @RN19]. Clustering implements UMAP for dimensionality reduction and HDBSCAN for clustering [@RN22; @RN21]. Tuning is supported via the Davies-Bouldin Index and the Calinski-Harabasz Index from scikit-learn [@RN19]. In brief, the Davies-Bouldin Index favors greater within-cluster similarity, while the Calinski-Harabasz Index favors greater separation between clusters [@RN24; @RN23]. Topic modeling relies on Gensim's LDA implementation, with cluster-to-label matching performed by computing cosine similarity between LDA topic words and predefined designators using BERT embeddings via Hugging Face Transformers; sentence-level representations are additionally available through SBERT [@RN27; @RN25; @RN26; @RN28]. The pipeline uses aggressive caching strategies to support iterative exploratory workflows where the same data may be processed repeatedly with varying parameters. Alternative vectorization methods, including SBERT, Word2Vec, GloVe, and LSA via TruncatedSVD, are also available [@RN29; @RN30; @RN31; @RN26].

# Research Impact

TACTIK has been instrumental in generating insights from diverse aviation text data [@RN33; @RN32]. Beyond published work on accident and incident data, it has also been applied to identify maintenance trends in aviation. The authors hosted a workshop at their university and plan future workshops and video tutorials to broaden adoption. The software includes a clearly documented pathway for community contributions.

# AI Usage Disclosure

Generative AI tools were used to assist with paper editing, bug fixing, and documentation of the code. All AI-assisted content has been reviewed and validated by the authors for technical accuracy and scholarly integrity.


# References
