### blog cools
https://nlpprogress.com/
https://www.ruder.io/
https://peltarion.com/blog/topic/nlp
article cool https://arxiv.org/pdf/2204.14264.pdf 
jsp https://medium.com/dair-ai/relaunching-the-nlp-newsletter-1bd91106c7da
https://thegradient.pub/
https://thegradient.pub/prompting/


## difficulties
- Language distribution / limited Data / curse of multilinguality
- Homogeneity / Lack of pre-training data 
- Over-representation of Western concepts
- Translation / Quality issues 
- Multilingual evaluation
- Dependence on retrieval : It assumes there is a single gold paragraph providing the correct answer and does not consider information from other paragraphs or pages.


## Citations
A number of factors has been found to be important for learning robust multilingual representations, including **shared tokens**, **subword fertility**, and **word embedding alignment**.

**Adapters** have been shown to improve robustness, lead to increased sample efficiency compared to fine-tuning

**tokenization** often produces poor segmentations for languages with a rich morphology or limited data.

Architeture of models can be adapted to incorporate information about morphology such as in the KinyaBERT model for Kinyarwanda

---
# preprocess
https://github.com/explosion/spaCy/tree/v3.5.0

# Benchmarks / normalized state-of-the-art performance
[list of some benchmarks](https://tmmtt.medium.com/natural-language-processing-nlp-dc2c1d8d4110)

### Multitask multilang benchmarks :  
[XTREME](https://sites.research.google/xtreme) 
[GLue / superGlue](https://gluebenchmark.com/)
XNLI
--[glge]--
--[TyDi QA](https://ai.google.com/research/tydiqa).--

### mono task / mono lang benchmarks :
[NLP-progress](http://nlpprogress.com/english/dependency_parsing.html)
[Tatoeba]
XNLI, XQuAD, and XCOPA are based on translations of established English datasets.



---
## Models  

## ByT5 | 300M-12.9B
08/03/2022
**Towards a token-free future with pre-trained byte-to-byte models**
https://arxiv.org/abs/2105.13626
the number of encoder layers is 3x more than the decoders.
ByT5 out-performs mT5 in most multilingual tasks, and especially for smaller models or when dealing with misspelled or noisy data, and is 50-100% faster.
![[Pasted image 20230205133314.png]]
![[Pasted image 20230205123454.png]]


## Turing ULR (XLM-E) | 279M-2.2B
19/04/2022
#1 at XTREME and GLue benchmark
https://arxiv.org/abs/2106.16138
Based on XLM from 2019 in [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)
and from [ELECTRA-style tasks](https://openreview.net/forum?id=r1xMH1BtvB) to cross-lingual language model pre-training.

two transformers encoders :
-   G the generator, like Bert, rained with the masked language modeling (MLM)
-   D the discriminator, who takes the result of G and determine which token is corrupted.
final loss is L = Lg + λLd
**![](https://lh4.googleusercontent.com/bvWcnGAW17znGP5dqeSvLMvSMTs1yxfHPtkKuTyeb3nToYhIyQFwF8nAinTv8C3OF_pRCkrnaHDtqhezVcLhmzg1e-ThiJMO2o0h_tQqMKnkj_XvlR6OJV-d7d1uGiitsDqufIsq184Hvvp3jeo0_Mg)**
**![](https://lh3.googleusercontent.com/VkqTavnxote5GPaMmCoIQzR6ymruErZ4wtacp8NeENUVppZhNbRujqHT03hOqY1sZEcvS1LHgrC4clBX9mWgnIAZtGYmUaJNgssfzuTfiWhpU46wdNvKBJT-yi71yOq3o5355O5zT8itElRlD7FCgek)**


## Vega
#1 at SuperGlue
https://arxiv.org/pdf/2212.01853.pdf
self-evolution learning for PLMs to wisely predict the informative tokens that should be masked, and supervise the masked language modeling (MLM) process with rectified smooth labels.
**![](https://lh6.googleusercontent.com/Fcc1CLwnYKPgeVPFUPFI7nvsq2ZRPR8a5s6L-8Mw84U0X13O7cBluBPUuY6K09mmBmOajjOwLdFJ1DtBg3_ico7wYxlfTxsH3gkrhqu_FjhYYCpRjXlBRb5tUNFhs59-j8HmbvqulUbAtM1-DAcBTVY)**
![](https://lh6.googleusercontent.com/BCsG_YZDuGdgchkQjicwx8OJ_P50vRiNqsDNNo0iYLR1TbYk02Zk3yQ2R1kIKP9DPZiXXhRB8cqEbvLihMfgMn6qqK9nq2JTAsec7qcDcC7uWd023-HbTsT7tAwaANgxSclhMQRWZm8FnFBP-G86zaQ)
## Polyglot
04/12/2022
https://arxiv.org/abs/2204.14264
6 tasks, namely topic classification, sentiment classification, named entity recognition, question answering, natural language inference, and summarization, covering 24 datasets and 49 languages
![](https://lh3.googleusercontent.com/ZOkS5H_evIN2JT5p8PEkwcjWilWk952uGQ_ym6kD9C6FJvqZNEcfdXCOTjxEW1KunELifW9Mj-933aXZkkDema-ib-i1Mgcvd4Yfz9OsOPDn2Mj1-RwV_nV_wfUKVVHrsRg24Q5nWTU_UCSyDDTIMi8)
[Bloom]
## MT5 / MT0 | 300m - 13B 
11/03/2021
https://arxiv.org/abs/2010.11934

## MBERT | 180B 
11/03/2021
https://arxiv.org/abs/2010.11934

## Bloom | Blommz 176B 
11/12/2022
https://arxiv.org/abs/2211.05100
https://arxiv.org/abs/2211.01786

# Token free models
## CANINE
CANINE is the first token- and vocabulary-free model, based on a hashing and downsampling strategy to work directly on the characters as Unicode code points.
He was trained on the TyDI QA dataset and outperformed other multilingual models, such as mBERT while having no predefined tokenization and 28% fewer parameters.

## Perceiver and Perceiver IO
The perceiver operates directly on the raw byte representation of the input. This enables the models to operate (more or less) on any type of data, be it text, images, point cloud, audio, etc.
He takes inspiration from the ByT5 paper to operate directly on the raw byte representation (UTF-8 for text) but extends it to multiple modalities.

## Charformer
Charformer consists of two parts: a dynamic, fast, and flexible method to learn subword representations automatically from n-grams and a model that incorporates it. By grouping n-characters together (n-gram) there is an increased opportunity to learn multiple representations of a word that may be more advantageous. Instead of using only one representation of subwords of a single character, the model can select the most informative representation of a word, by weighting multiple representations from the different n-grams. These are then downsampled in groups of 2 with mean pooling to get a sequence with a shorter length.

This module is called Gradient-Based Subword Tokenization (GBST) and is the token-free module used by the Charformer. Since all components in the module are pre-defined, except for how to weight/score each n-gram representation, this can be done efficiently and quickly. Also, since the scoring is done using the Softmax function it is also differentiable and learnable. This means that better text representations can update on new vocabulary or languages dynamically.

![](https://images.prismic.io/peltarionv2/f83f7da0-edb4-4d77-9313-e60bcf6aa110_Charformer+GBST+model.png?auto=compress%2Cformat&rect=0%2C0%2C1951%2C1711&w=1300&h=1140)

Creating an n-gram of characters shortens the length of the text by n. For instance, the text “Successfully” as a 4-gram would be “Succ”, “essf”, “ully”; that is 4 times shorter than the original text. Therefore, these n-grams are mean pooled and repeated X number of times to be the same length again. The pooled and duplicated embedding for “succ” in the image below would be C4, 1, “essf” C4, 2 , and “ully” C4, 3. These are then scored via a weighting and mean pooled to a shorter representation. Since pooling removes the position of tokens, position embeddings are added to the tokens at each pooling step.

Charformer performs on par or outperforms the regular T5 on multiple English tasks and outperforms both ByT5 and CANINE while being smaller, faster, and with shorter sequences. Unlike CANINE, a model using the GBST, s.a. Charformer is interpretable in how the tokens are represented. Charformer is as of this writing the current State-of-the-Art (SOTA) method when it comes to token-free models. For those interested in learning more about the model, I highly recommend this [short and pedagogical video](https://www.youtube.com/watch?v=debgj24BAZE).


--[MShenNonG+TDT]()--
--[CoFe]()--
[Palm]
[Hyper-X: A Unified Hypernetwork for Multi-Task Multilingual Transfer]()
[CORA]




---
## Articles
Adaptaters : [adaptaters](https://medium.com/dair-ai/adapters-a-compact-and-extensible-transfer-learning-method-for-nlp-6d18c2399f62)
### State of art
[token free research](https://peltarion.com/blog/data-science/towards-a-token-free-future-in-nlp)

[A deep dive into multilingual NLP models](https://peltarion.com/blog/data-science/a-deep-dive-into-multilingual-nlp-models)
[ACL 2022: Association for computational linguistic](https://www.ruder.io/acl2022/)
[The State of Multilingual AI](https://www.ruder.io/state-of-multilingual-ai/)
[Challenges and Opportunities in NLP Benchmarking](https://www.ruder.io/nlp-benchmarking/)

### Multitask
[How to Create and Train a Multi-Task Transformer Model](https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240)

### Mulitlang
[Multi-domain Multilingual Question Answering](https://www.ruder.io/multi-qa-tutorial/)
[How to Multi-Task in Multiple Languages with the mT5 Transformer](https://towardsdatascience.com/going-global-how-to-multi-task-in-multiple-languages-with-the-mt5-transformer-892617cd890c)

[Many Languages, One Deep Learning Model](https://towardsdatascience.com/many-languages-one-deep-learning-model-69201d02dee1)
[ACL 2022 Limited Data Learning Tutorial](https://github.com/diyiy/ACL2022_Limited_Data_Learning_Tutorial)
[ACL 2022 Tutorial: Zero- and Few-Shot NLP with Pretrained Language Models](https://github.com/allenai/acl2022-zerofewshot-tutorial)




---
## Paper
### Datasets
[Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00447/109285/Quality-at-a-Glance-An-Audit-of-Web-Crawled)
[Natural language processing: state of the art, current trends and challenges](https://link.springer.com/article/10.1007/s11042-022-13428-4)  

### QA
[Towards More Equitable Question Answering Systems: How Much More Data Do You Need?](https://aclanthology.org/2021.acl-short.79.pdf)
[Applying Multilingual Models to Question Answering (QA)](https://arxiv.org/pdf/2212.01933.pdf)

### Multitask
[Multi Task Learning For Zero Shot Performance Prediction of Multilingual Models](https://arxiv.org/abs/2205.06130)
[Multi-Task Deep Neural Networks for Natural Language Understanding - 30/05/19](https://arxiv.org/abs/1901.11504v1)

### Multilang
[Towards Afrocentric NLP for African Languages: Where We Are and Where We Can Go](https://aclanthology.org/2022.acl-long.265.pdf)
[KinyaBERT: a Morphology-aware Kinyarwanda Language Model](https://aclanthology.org/2022.acl-long.367.pdf)

[Expanding Pretrained Models to Thousands More Languages via Lexicon-based Adaptation](https://aclanthology.org/2022.acl-long.61/)

[Cross-Lingual Ability of Multilingual BERT: An Empirical Study](https://openreview.net/forum?id=HJeT3yrtDr)
[Investigating Cross-Lingual Alignment Methods for Contextualized Embeddings](https://aclanthology.org/K19-1004.pdf)

### Zero shot learning
[Multi Task Learning For Zero Shot Performance Prediction of Multilingual Models 2021](https://aclanthology.org/2022.acl-long.374/)
[Cross-Lingual BERT Transformation for Zero-Shot Dependency Parsing](https://aclanthology.org/D19-1575.pdf)
[BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting](https://arxiv.org/abs/2212.09535)


---
## Others  

### multimodal  
[Mu2slam - 19/12/22](https://arxiv.org/abs/2212.09553)
[m3p - 02/11/21 - 56cit](https://ieeexplore.ieee.org/document/9577347)
[MURAL - 11/21](https://aclanthology.org/2021.findings-emnlp.293/)


### Cool link
https://www.ruder.io/tag/natural-language-processing/

### archive
[Meta-Learning for Effective Multi-task and Multilingual Modelling](https://arxiv.org/abs/2101.10368)
