# shopee-product-matching

Implementaion of [Kaggle Competition](https://www.kaggle.com/c/shopee-product-matching)

## Problem Definition

Retrieve same product using given images and texts per each product(below are examples)

![](https://www.researchgate.net/profile/Artsiom-Sanakoyeu/publication/333815726/figure/fig2/AS:770621805977600@1560741964440/Qualitative-image-retrieval-results-on-Stanford-Online-Products-33-We-randomly-choose.ppm)

## How to solve

### train dataset analysis 
- 34250 image & text information
- total 11014 label_group(classes)
- 2 ~ 51 same class instance num( class imbalance)
- It's **hard to solve as Classification model!**

### Using Metric Learning

Basic Concept

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdpuky0%2FbtqIjeVyxZo%2FSnmmbKkMGT6aD1JSWybngk%2Fimg.png)

Feature Extraction from two images

![](https://www.mdpi.com/symmetry/symmetry-10-00385/article_deploy/html/images/symmetry-10-00385-g001.png)
- It can just few examples in one mini-batch(calculate mini-batch size times) -> how to get more pair features?

Using in-batch constrastive loss

![In-batch constrastive learning](./imgs/in_batch_contrastive_learning.png)
- It calculate all relations in one mini-batch(calculate mini-batch size * mini-batch size times)

## To-do

## References
