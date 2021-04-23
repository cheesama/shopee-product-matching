# shopee-product-matching

Implementaion of [Kaggle Competition](https://www.kaggle.com/c/shopee-product-matching)

## Problem Definition

Retrieve same product using given images and texts per each product(below are examples)

![](https://www.researchgate.net/profile/Artsiom-Sanakoyeu/publication/333815726/figure/fig2/AS:770621805977600@1560741964440/Qualitative-image-retrieval-results-on-Stanford-Online-Products-33-We-randomly-choose.ppm)

## How to solve

### train dataset analysis(no valid dataset provided)
- 34250 image & text information
- total 11014 label_group(classes)
- 2 ~ 51 same class instance num( class imbalance)
- It's **hard to solve as Classification model!**

### Using Metric Learning

1) Basic Concept
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdpuky0%2FbtqIjeVyxZo%2FSnmmbKkMGT6aD1JSWybngk%2Fimg.png)

![](https://kevinmusgrave.github.io/pytorch-metric-learning/imgs/contrastive_loss_similarity_equation.png)

appropriate values would be **pos_margin = 1 and neg_margin = 0**, in case of cosine similarity

---

2) Feature Extraction from two images
![](https://www.mdpi.com/symmetry/symmetry-10-00385/article_deploy/html/images/symmetry-10-00385-g001.png)
- It can handle just few examples in one mini-batch(calculate mini-batch size times) -> how to get more pair features?

---

3) Using in-batch constrastive loss
![In-batch constrastive learning](./imgs/in_batch_contrastive_learning.png)
- It calculate all relations in one mini-batch(calculate mini-batch size * mini-batch size times)
- But still there are a lot of pairs information outside of bathes, how to expand it?

---

4) Using external batch output when comparing current batch
![](https://i.ytimg.com/vi/SDKDSvv9oTk/maxresdefault.jpg)
- It has internal batch output queue which store latest batches & upate
- Model can watch previous batch output from via queue and use it to calcluate pair distances

---

5. Each Batch have both soft & hare pairs. it is proper to apply same margin?
![multi_similariry_pair_mining](./img/multi_similariry_pair_mining)
- This mining use cosine similariy as distance metric
- Negative pairs are chosen if they have similarity greater than the hardest positive pair, minus pre-defined margin
- Positive pairs are chosen if they have similarity less than the hardest negative pair, plus pre-defined margin

## To-do
- compound Text Feaure(TF-IDF of DistilBERT ...)
- add external dataset([deepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), [SOP](https://cvgl.stanford.edu/projects/lifted_struct/) ...)
- hyper-parameter tuning
- submission to kaggle

## References
[Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf)

[pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)
