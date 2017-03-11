# Tree-based Gated Recurrent Units

This repository aims to be my repository that I will store the scripts for my master thesis.

The content of this repository is the implementation (in tensorflow) of an alteration of [this paper](https://arxiv.org/pdf/1503.00075.pdf) written by Kai Sheng Tai, Richard Socher, Christopher D. Manning. My implementation is the hybrid between a recursive neural networks and Gated Recurrent Units. Moreover it is important to mention that this alteration is tested
on sentiment classification , both binary and fine-grained. 

I am testing two different alterations  of this idea. 

## 1rst alteration 

![alt tag](https://github.com/VasTsak/master_thesis/blob/master/rersive_gru/first_approach.png?raw=true)

## 2nd alteration 

![alt tag](https://github.com/VasTsak/master_thesis/blob/master/rersive_gru/second_approach.png?raw=true)

In order to run those two models, you should run the following script. Which downloads the pre-trained word embeddings.

./fetch_and_preprocess.sh
