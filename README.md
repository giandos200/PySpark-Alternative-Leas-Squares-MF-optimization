# PySpark Alternating Least Squares implemetation with K-core preprocessing, different Splitting strategies and Evaluation metrics
In this repository we reproduce the model proposed by Koren et al. ["Matrix Factorization Techniques for Recommender Systems"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5197422), IEEE 2009. 
We implement a classic Recommender Systems Pipeline, i.e., from item filtering (K-core) to model training-evaluation and recommendation in a PySpark scalable procedure.

# Alternating Least Squares (ALS)
Alternative Least Squares is a matrix-factorization optimization techniques to train sparse User-Item Rating matrix for scalability improvements.
Let be:
- $\mathbf{x}_u \in \mathbf{R}^k$ a $k$ dimensional vectors as a latent representation of the user $u$;
- $\mathbf{y}_i \in \mathbf{R}^k$ a $k$ dimensional vectors as a latent representation of the item $i$;
- $\hat{r}_{ui} = \mathbf{x}_u^T \mathbf{y}_i$ the predicted user-item rating;
- $\min_{x,y} \sum_{u,i} (r_{u,i}-\hat{r}_{u,i})^{2} + \lambda(|| \mathbf{x}_u ||^2 + || \mathbf{y}_i ||^2)$ be the objective function;

ALS techniques rotate between fixing the $x_u$'s and fixing $y_i$'s.  When all $y_i$'s are fixed, the system recomputes the $x_u$’s by solving a least-squares problem, and vice versa. This ensures that each step decreases loss until convergence;

### Sampling strategies
We implemented three type of samling strategies:
- 'Interaction' which chose a percentage % of the interactions;
- 'User' which chose a percentage % of interaction for each user;
- 'Item' which chose a percentage % of interaction for each item;

### K-core Filtering
We iteratively filter the user and item with less than $K$ interactions until for each item and user we have a number of interaction $>K$.

### Splitting
We implemented different Train/Test Splitting strategies for Training and Evaluating the Recommendation models
- 'Hold-Out' dividing in train and test set based on all the interactions
- 'User-Hold-Out' dividing in training and test set based on singular User interactions
- 'Leave-One-Out' putting a random item of each user in the test set

### Training and Evaluation
We trained the ALS model available in pyspark.ml.recommendation on the train set, and we evaluate the performance on the test set on different metrics and different Top-N list of items recommended:
- <img src="https://latex.codecogs.com/svg.image?RMSE&space;=&space;\sqrt[2]{\left(&space;\frac{1}{|\mathbf{R}|}&space;\sum_{\hat{r}_{ui}\in&space;\mathbf{R}}(r_{ui}-\hat{r}_{ui})\right)&space;}"/>
- $NDCG$
- $Precision$
- $Recall$
- $MeanAveragePrecision$


### Installation guidelines:
Following, the instruction to install the correct packages for running the experiments (numba==0.48.0 is mandatory)

```bash
$ python3 -m venv venv_ALS
$ source venv_ALS/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Training and test the model
To train and evaluate ALS Recommender Systems with all the metrics, you may run the following command:

```bash
$ python -u main.py
```

### Results

- $RMSE = 1.0280$

|   TopN |       NDCG |   Precision |      Recall |   MeanAveragePrecision |
|--------|------------|-------------|-------------|------------------------|
|      1 | 0.00632911 |  0.00632911 | 0.000205189 |             0.00632911 |
|      2 | 0.011226   |  0.0126582  | 0.0010648   |             0.00791139 |
|      3 | 0.0127501  |  0.014346   | 0.0015878   |             0.00752461 |
|      4 | 0.0151187  |  0.0174051  | 0.00313034  |             0.00772679 |
|      5 | 0.0153556  |  0.0172152  | 0.0034662   |             0.00712975 |
|     10 | 0.0230513  |  0.0267089  | 0.0104655   |             0.00718967 |
|     20 | 0.0300045  |  0.0313291  | 0.0245317   |             0.00706743 |
|     30 | 0.0353195  |  0.0331224  | 0.0377007   |             0.00736266 |
|     40 | 0.0422182  |  0.0351266  | 0.0546082   |             0.00805145 |
|     50 | 0.0487761  |  0.0356456  | 0.0713498   |             0.00874634 |
|     60 | 0.05544    |  0.0363924  | 0.0871217   |             0.00950496 |
|     70 | 0.0626016  |  0.0366908  | 0.104368    |             0.010354   |
|     80 | 0.069045   |  0.0368354  | 0.119117    |             0.011221   |
|     90 | 0.0760735  |  0.0371167  | 0.136286    |             0.0121385  |
|    100 | 0.0824132  |  0.0370886  | 0.151604    |             0.0130137  |

### Example of Recommendation
We chose a random user and display a Top-10

- $user = 434$


|   userId |  title_new   | title                                                        |
| -------- | ------------ | -------------------------------------------------------------|
|      434 | 907          |  Candyman: Farewell to the Flesh (1995)                      |
|      434 | 874          |  Microcosmos: Le peuple de l'herbe (1996)                    |
|      434 | 928          |  Paradise Lost: The Child Murders at Robin Hood Hills (1996) |
|      434 | 843          |  Love! Valour! Compassion! (1997)                            |
|      434 | 847          |  Something to Talk About (1995)                              |
|      434 | 687          |  Fallen (1998)                                               |
|      434 | 557          |  Bad Boys (1995)                                             |
|      434 | 913          | Miserables, Les (1995)                                       |
|      434 | 1028         |  Braindead (1992)                                            |
|      434 | 688          |  Gigi (1958)                                                 |

### Colab Notebooks
You can run the experiments at this [link](https://colab.research.google.com/drive/1o18KCbRiM3xtNwtbCYw-_pdM47vdqzyO?usp=sharing).

### Contributors
- [Giandomenico Cornacchia](https://github.com/giandos200)
- [Daniele Malitesta](https://github.com/danielemalitesta)

