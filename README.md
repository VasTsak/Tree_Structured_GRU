# master_thesis

This repository aims to be my repository that I will store the scripts for my master thesis.

The content of this repository is the implementation of an alteration of [this paper](https://arxiv.org/pdf/1503.00075.pdf) written by Kai Sheng Tai, Richard Socher, Christopher D. Manning. My implementation is the hybrid between a recursive neural networks and Gated Recurrent Units.

I am testing two different alterations  of this idea. 

## 1rst alteration 

![alt tag](https://raw.github.com/VasTsak/master_thesis/blob/master/rersive_gru/first_approach.png)

## 2nd alteration 

\begin{gather}
z_{jk} =\sigma ( W^{z}x_{j} +\displaystyle\sum\limits_{l=1}^{N} U^{z}_{kl}h_{jl} + b^{z})
\end{gather}

\begin{gather}
r_{jk} =\sigma ( W^{r}x_{j} +\displaystyle\sum\limits_{l=1}^{N} U^{r}_{kl}h_{jl} + b^{r})
\end{gather}

\begin{gather}
\tilde{h}_{j} = \tanh (W^{h}x_{j} +\displaystyle\sum\limits_{l=1}^{N} U^{h}_{l}(h_{jl} \odot r_{jl}) + b^{u})
\end{gather}

\begin{gather}
h_{j} = \displaystyle\sum\limits_{l=1}^{N} (\dfrac{1-z_{jk}}{N})  \odot \tilde{h}_{j}  + \displaystyle\sum\limits_{l=1}^{N} \dfrac{z_{jk}}{N} \odot h{jk}
\end{gather}


If you feel like you would like to contribute to it or suggest something please do it. 

Cheers !
