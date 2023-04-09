| Modèle  | date | Comment | nb Param |
|:---|:---:|:---:|:---:|
|  **Polyglot**|04/12/2022 | built on MT5 -> prompt| 580M |
|  **Vega** |19/04/2022 | #1 at SuperGlue | 6B |
|  **XLM_E** |19/04/2022 | #1 at XTREME and GLue benchmark|279M-2.2B|
|  **Byt5** |08/03/2022 |  ByT5 out-performs mT5 in most multilingual tasks, and especially for smaller models or when dealing with misspelled or noisy data, and is 50-100% faster.| 300M-12.9B|
|  MT6 |04/09/2021|   |⨯  |
|  DeltaLM |25/06/2022|   ~ | ~|  | ~|
|  **Charformer**| 23/06/2021 |  T5 with n-gramm => outperform ByT5 and T5  |~  |  
|  **MT5** |11/03/2021| ~||300M-12.9B |
|  MBert|24/05/2019||


--- 
## Polyglot | 580M
04/12/2022
https://arxiv.org/abs/2204.14264
6 tasks, namely topic classification, sentiment classification, named entity recognition, question answering, natural language inference, and summarization, covering 24 datasets and 49 languages
![](../_ressources/Pasted%20image%2020230206133902.png)
![](../_ressources/Pasted%20image%2020230207152023.png)
![](https://lh3.googleusercontent.com/ZOkS5H_evIN2JT5p8PEkwcjWilWk952uGQ_ym6kD9C6FJvqZNEcfdXCOTjxEW1KunELifW9Mj-933aXZkkDema-ib-i1Mgcvd4Yfz9OsOPDn2Mj1-RwV_nV_wfUKVVHrsRg24Q5nWTU_UCSyDDTIMi8)
- PolyPrompt can achieve improvement, especially with the introduction of **high-resource datasets, as same in the cross language zero-shot tranfer**

- Dataset Perspective: the strengths of  PolyPrompt in the co-occurring languages on different datasets are inconsistent due to dataset bias. For example, en, de, and fr co-occur on  PAWS-X and XNLI. PolyPrompt **was better in the  short sentence.**
- Model Perspective: PolyPrompt achieves  overall performance improvements on the 7 target  datasets, but it cannot perform well on all samples (e.g., **worse performance on long sentences**).
- PolyPrompt is **worse at  handling long question samples**
- It is difficult for PolyPrompt to bring gains for languages that appear only once in the 7 target datasets unless high-resource datasets are introduced. 
  For example,  PolyPrompt showed a slight performance improvement over vanilla mT5 in languages bn, fi, id,  and te that only appeared in the TyDiQA dataset.  
  When introducing high-resource English datasets,  the performance of bn is significantly improved especially for long context and short answers samples, while the performance improvement of fi,  id, and te is still limited until a high-resource multilingual training dataset PANX is introduced.


--- 
## Vega | 6B
#1 at SuperGlue
https://arxiv.org/pdf/2212.01853.pdf
self-evolution learning for PLMs to wisely predict the informative tokens that should be masked, and supervise the masked language modeling (MLM) process with rectified smooth labels.
+Transductive Fine-tuning + Prompt-Tuning + Adversarial Fine-Tuning
![](../_ressources/Pasted%20image%2020230206133846.png)
![](../_ressources/Pasted%20image%2020230206133851.png)
Vega v2 significantly surpasses the powerful human baselines in terms of average score. and achieves state-of-the-art performance on four (relatively) low-resource tasks, i.e.,  CB, COPA, RTE, and WSC. 
We attribute this success to the novel self-evolution learning mechanism and KD-based prompt transfer method. More specifically, the former enhances Vega v2’s ability to extract informative patterns, while the latter alleviates the problem of overfitting and boosts the model  
performance on low-resource tasks.  

In addition, compared to the other larger PLMs, e.g., PaLM (Chowdhery et al., 2022b) which consists  of 540 billion parameters, our 6-billion-parameter Vega v2 can achieve competitive or even better performance on the SuperGLUE benchmark. This inspires us to conclude that scaling PLMs to larger

---
## Byt5 | 300M-12.9B 
08/03/2022  
https://arxiv.org/abs/2105.13626  
**Towards a token-free future with pre-trained byte-to-byte models**  
the number of encoder layers is 3x more than the decoders.  
ByT5 out-performs mT5 in most multilingual tasks, and especially for smaller models or when dealing with misspelled or noisy data, and is 50-100% faster.

### Xtreme 
![](../_ressources/Pasted%20image%2020230206133817.png)
### Glue / super Glue
![](../_ressources/Pasted%20image%2020230207095418.png)
### Generative tasks
![](../_ressources/Pasted%20image%2020230209111646.png)
### QA 
![](../_ressources/Pasted%20image%2020230207095658.png)
### Multilangual Multitask

#### Pro
- at model sizes under 1 billion parameters,  
- on generative tasks, 
-  on multilingual tasks  with in-language labels, 
-  on word-level tasks sensitive to spelling and/or pronunciation
-  in the presence of various types of noise.

#### Cons
- English classification tasks for model  sizes over 1 billion parameters.
it will also be important to evaluate token-free approaches on a more diverse set of tasks, especially those where character-based models have  traditionally struggled (word similarity tasks, syntactic and semantic tagging task and trasnlate non en to en)


---
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
![](https://lh4.googleusercontent.com/bvWcnGAW17znGP5dqeSvLMvSMTs1yxfHPtkKuTyeb3nToYhIyQFwF8nAinTv8C3OF_pRCkrnaHDtqhezVcLhmzg1e-ThiJMO2o0h_tQqMKnkj_XvlR6OJV-d7d1uGiitsDqufIsq184Hvvp3jeo0_Mg)
![](../_ressources/Pasted%20image%2020230206133829.png)
![](../_ressources/Pasted%20image%2020230206133841.png)**![](https://lh3.googleusercontent.com/VkqTavnxote5GPaMmCoIQzR6ymruErZ4wtacp8NeENUVppZhNbRujqHT03hOqY1sZEcvS1LHgrC4clBX9mWgnIAZtGYmUaJNgssfzuTfiWhpU46wdNvKBJT-yi71yOq3o5355O5zT8itElRlD7FCgek)**
![](../_ressources/Pasted%20image%2020230208093829.png)
** pas si bob syr TydiQA**

---

## Charformer | 300m - 13B 
23/02/2022
https://arxiv.org/pdf/2106.12672.pdf
Charformer consists of two parts: a dynamic, fast, and flexible method to learn subword representations automatically from n-grams and a model that incorporates it. By grouping n-characters together (n-gram) there is an increased opportunity to learn multiple representations of a word that may be more advantageous. 

Charformer performs on par or outperforms the regular T5 on multiple English tasks and outperforms both ByT5 and CANINE while being smaller, faster, and with shorter sequences. Unlike CANINE, a model using the GBST, s.a. Charformer is interpretable in how the tokens are represented. Charformer is as of this writing the current State-of-the-Art (SOTA) method when it comes to token-free models. For those interested in learning more about the model, I highly recommend this [short and pedagogical video](https://www.youtube.com/watch?v=debgj24BAZE).

#### Monolingual
![](../_ressources/Pasted%20image%2020230206143620.png)
#### Multilingual
![](../_ressources/Pasted%20image%2020230207154315.png)
![](../_ressources/Pasted%20image%2020230207154229.png)

---
## MT5 | 300m - 13B 
11/03/2021
https://arxiv.org/abs/2010.11934
![](../_ressources/Pasted%20image%2020230208095227.png)


---
