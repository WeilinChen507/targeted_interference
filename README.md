# CODE

official code of Doubly Robust Causal Effect Estimation on Networked Data via
Targeted Learning [[paper]](https://icml.cc/virtual/2024/oral/35570)

# Cite this paper

Chen, W., Cai, R., Yang, Z., Qiao, J., Yan, Y., Li, Z. &amp; Hao, Z.. (2024). Doubly Robust Causal Effect Estimation under Networked Interference via Targeted Learning. <i>Proceedings of the 41st International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 235:6457-6485

```
@InProceedings{pmlr-v235-chen24c,
  title = 	 {Doubly Robust Causal Effect Estimation under Networked Interference via Targeted Learning},
  author =       {Chen, Weilin and Cai, Ruichu and Yang, Zeqin and Qiao, Jie and Yan, Yuguang and Li, Zijian and Hao, Zhifeng},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {6457--6485},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/chen24c/chen24c.pdf},
  url = 	 {https://proceedings.mlr.press/v235/chen24c.html},
  abstract = 	 {Causal effect estimation under networked interference is an important but challenging problem. Available parametric methods are limited in their model space, while previous semiparametric methods, e.g., leveraging neural networks to fit only one single nuisance function, may still encounter misspecification problems under networked interference without appropriate assumptions on the data generation process. To mitigate bias stemming from misspecification, we propose a novel doubly robust causal effect estimator under networked interference, by adapting the targeted learning technique to the training of neural networks. Specifically, we generalize the targeted learning technique into the networked interference setting and establish the condition under which an estimator achieves double robustness. Based on the condition, we devise an end-to-end causal effect estimator by transforming the identified theoretical condition into a targeted loss. Moreover, we provide a theoretical analysis of our designed estimator, revealing a faster convergence rate compared to a single nuisance model. Extensive experimental results on two real-world networks with semisynthetic data demonstrate the effectiveness of our proposed estimators.}
}
```

# Thanks
Our code partly follows the code from [[RRNet]](https://github.com/DMIRLAB-Group/RRNet) and [[NetEst]](https://github.com/songjiang0909/Causal-Inference-on-Networked-Data). Thanks for their code!
