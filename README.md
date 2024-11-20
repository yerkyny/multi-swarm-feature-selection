This project implements a novel feature selection algorithm, namely a Multi-swarm Binary Particle Swarm Optimization with local search and random regrouping.

## Content
* [Introduction](#intro)
    1. [Wrapper methods]
    2. [Particle swarm optimization]
    3. Binary PSO
* [Multi-swarm binary PSO with random regrouping and local search](#msbpso)
    1. [Personal best update]
    2. [Local search mechanism]
    3. [Final algorithms]
* [Use example](#example)
* [References](#references)

## <a class="intro" id="intro">Introduction</a>
### 1. Wrapper methods
In literature on feature selection (FS) "wrapper methods" refer to the class of feature selection techniques that evaluate the usefulness of subsets of features based on validation performance of a specific machine learning model. A wrapper FS algorithm involves a search strategy that defines the order in which feature subsets are evaluated. Given a dataset with $m$ features, we have $2^m$ subsets in total and it is typically infeasible to assess every subset in a reasonable time period.

Simplest search strategies for wrapper methods include

* Stepwise forward selection: starting from an empty feature subset, at each step add the feature that gives the best accuracy improvement.
* Stepwise backward elimination: starting with the full feature set, at each step exclude the feature whose loss is the least significant.
* Bidirectional search: a combination of forward selection and backward elimination, testing at each step for variables to be included or excluded.
    
Forward and backward search need to assess $m(m-1)/2$ subsets unless we fix the maximal/minimal number of features that we want to keep. This may still be computationally too expensive and both algorithms are prone to stagnation in "local optima" due to their greedy top-down approach.
Bidirectional search needs to assess around $O(m*n_{steps})$ subsets. It is much more flexible, especially when randomized and allowed to add or eliminate several features at one step, but then it becomes much more sophisticated.

A well recognized approach to search strategies for feature selection is *swarm intelligence*. In the context of optimization, it is a family of metaheuristic optimization algorithms designed to optimize nonconvex functions. They are inspired by the collective behavior of social organisms, such as birds flocking, ants foraging for food, and bees locating optimal flowers. These algorithms model the simplified social processes and mechanisms such as following the leader, avoiding crowds, hunting, etc., and are usually stochastic to some extent.
### 2. Particle Swarm Optimization
Particle Swarm Optimization (PSO) is one of the simplest and earliest swarm intelligence algorithms. It represents potential solutions as positions of particles in the search space $x_1(t),\dots,x_n(t) \in \mathbb{R}^m$. Their movement is governed by individual experience, collective knowledge of the swarm and the momentum. The optimized function $L(x)$ will be called the cost function. The algorithm works as follows:
1. Initialize the positions of particles $x_1(0),\dots,x_n(0) \in \mathbb{R}^d$ and their velocities $v_1(0),\dots,v_n(0) \in \mathbb{R}^d$. Set the velocity update weights $w, c_1, c_2 \geq 0$
2. For each step $t=0, 1, 2, \dots$
    * Calculate the costs $L(x_i(t)), \hspace{6pt} i=1,\dots,n$
    * Update the best personal position if each particle and the global best position of each of the swarm.
      
        $$pbest_i(t) = x, \hspace{6pt} s.t. \hspace{6pt} L(x) = \min_{s\leq t} L(x_i(s))$$
      
        $$gbest_i(t) = x, \hspace{6pt} s.t. \hspace{6pt} L(x) = \min_{1\leq i\leq n} L(pbest_i(t))$$
    * Update velocities and positions as follows:
      
        $$v_i(t+1) = w v_i(t) + c_1 r_i^1(t) (pbest_i(t)-x_i(t)) +c_2 r_i^2(t) (gbest_i(t)-x_i(t))$$
      
        $$x_i(t+1) = x_i(t) + v_i(t+1),$$
      
        where $r_i^1, r_i^2$ are i.i.d. random variables with uniform distribution $U(0,1)$.
The weights $w, c_1, c_2$ control the stregth of momentum, attraction to personal best and to global best positions.

### 3. Binary PSO
In case of feature selection, we want to optimize the the validation error in the space of binary vectors $\{0,1\}^d$. Subsets of the total feature set $\{X^1,\dots,X^d\}$ are encoded as

$$ S \mapsto x = (x^1,\dots,x^d), \text{ where } \hspace{6pt} x^j = 1 \hspace{6pt} \text{ iff } \hspace{6pt} X^j\in S.$$

The binary PSO (BPSO) algorithm differs from the continuous space version only in the position update step: on every step we generate i.i.d. rv.s $r^1,\dots, r^d \sim U(0,1)$ and update the positions as below.

$$ x_i(t+1) = 1 \hspace{6pt} \text{ if } \hspace{6pt} r^i < (1+e^{-v_i(t)})^{-1},$$

$$ x_i(t+1) = 0 \hspace{6pt} \text{ otherwise}, $$

The advantages of PSO are that it doesn't requires differentiability of the cost function, it is easy to implement and parallelize and it has chance to escape local minima. The common issues wit PSO are
* Premature convergence: the algorithm often converges to quickly without singnificantly reducing the cost
* Stagnation of the global best position, as the leader does not learn from other particles
* Other particles are dominated by the gbest
* The algorithm doesn't use any statistical information about the data. 
According to , Transformers rely on self-attention mechanisms.

There is a great variety of modifications of PSO (including binary version) that try to address the common issues of PSO. There are also many other algorithms multi-agent algorithms inspired by nature, such as Ant Colony Optimization (ACO) [[Dorigo et al. (2006)](https://ieeexplore.ieee.org/abstract/document/4129846?casa_token=JgeeCH5G2LcAAAAA:bDRukvnSA-DISDU3JQ5lqQCvrtKXTAl-Qq4nZ4qKNRIgn9QhaftbhjJfc_uEh5W20YWQ5k7mq8Hb)], Artificial Bee Colony (ABC) [[Karaboga et al. (2014)](https://link.springer.com/article/10.1007/s10462-012-9328-0)], Firefly Algorithm (FA) [[Yang (2009)](https://arxiv.org/pdf/1003.1466)].

## <a class="intro" id="msbpso">Multi-swarm binary PSO with random regrouping and local search</a>

The algorithm we propose is designed specifically for tree-based models. It uses the feature importances (Rf-like) produced by the model and the fact that decision trees tend to produce sparse solutions, by using only those features involved in the splits. The multi-swarm approach is inspired by the Dynamic multi-swarm particle swarm optimizer [Liang, Suganthan (2005)](https://ieeexplore.ieee.org/abstract/document/1501611?casa_token=X67JkuriAOsAAAAA:ajByfMa2Tsjil54yDXnbMbfFO9z8YOt1yeJKQ2IJqwdRIxbJ-Nbz0-OLyy8oL1ThH0PYmTecxW3f). The main borrowed idea is to divide the population of particles into multiple swarms, that search the space independently and every $T$ iterations of the algorithm the particles are randomly regrouped into new subswarms.\
Our heuristic argument in favor of the multi-swarm approach is following: while the vanilla PSO has one attractor (the gbest) that doesn't learn from other particles and tends to stagnate, the multi-swarm algorithm has several attractors that will very likely change in a fixed amount of steps, i.e when the random regrouping happens. This way the algorithms converges slower and has more chance to explore more distant regions.

The leader of the subswarm is the particle that holds the gbest, it performs the local search in every step. Within every swarm there are $n_{extra}$ extra searchers. These are the particles (except the leader of the subswarm) that can perform the local search with a specified probability $p$ at each step, and with probability $1-p$ they execute the usual PSO update.

### 1. Personal best update
The update of pbest position of a particle is done when the current position has lower cost than the current pbest position, just like in usual PSO. But the updated position (subset) will only include the features that were actually used by the model. Tree-based models normally use less features than provided, depending on the number of splits they can make. It leads to much more sparse subsets. The cost of a position is the validation error of the model trained on the feature subset corresponding to that position.
### 2. Local search mechanism
During the local search the next position of a particle $x(t+1)$ is calculated according to a randomized rule. The rule uses the importances of the feature in the subset encoded bythe pbest position and the pairwise distances between the features. The distance between features $X^i, X^j$ and the total distance from $X^j$ to a feature subset $S$ are defined as
    
$$d(X^i,Xj) = 1 - |\rho(X^i, X^j)|,$$
    
$$d_{\text{total}}(S, X^j)^2 = \sum_{X^i \in S} d(X^i, X^j)^2,$$
    
where $\rho$ is the Spearman correlation. The total distance $d_{\text{total}}(S, X^j)$ is thus a metric of independence of the feature $X^j$ for $S$. Heuristically features with higher distance can bring more information about the target into the subset of features $S$. The position of the local search is computed as below:
* Let the pbest position of the particle $pbest_k(t)$ correspond to the feature subset $S$. We generate two independent rv.s $\xi\sim U(\{1,\dots,l\})$ and $\eta\sim U(\{1,\dots,\max(l, d-l)\})$
* $\xi$ features from $S$ with the lowest importance scores will be considered for removal, each candidate will be removed with probability $0.5$. $\eta$ features that are not included and have the highest total distance from the features in $S$ will be considered for addition, each candidate will be added with probability $0.5$.
Any feature in $S$ can be removed and any feature outside $S$ can be added, but features with higher importance are more likely to stay, and features with higher distance to $S$ are more likely to be added.
### 3. Final algorithm
Now we can summarize the algorithm.
1. Set the velocity update weights $w, c_1, c_2 \geq 0$, number of subswarms, the number of extra searchers $n_{extra}$, the probability of local search $p$ and the regrouping period $T$
2. Initialize the positions of particles $x_1(0),\dots,x_n(0) \in \mathbb{R}^d$ and their velocities $v_1(0),\dots,v_n(0) \in \mathbb{R}^d$. Group particles into subswarms
3. For each step $t=0, 1, 2, \dots$
    * Calculate the costs of all particles, update personal best positions and the global best position of every subswarm
    * Within each subswarm the leader performs the local search. The next $n_{extra}$ particles with lowest pbest costs perform the local search with probability $p$ or do a usual BPSO step. The rest of the particles do the usual BPSO step
    * If $t \equiv 0\text{ mod } T$, randomly regroup the subswarms.
4. Proceed until the stopping condition is reached.

## <a class="example" id="example">Use example</a>
In the example below we use our algorithm for binary classification of the Musk dataset (version 2) from the UCI repository. We show how to run the algorithms to optimize the recall metric of LightGBM classifiers


```python
import swarm_feature_selection
import numpy as np
import pandas as pd
import sklearn
import scipy
import lightgbm as lgbm
import logging
import matplotlib.pyplot as plt
```


```python
from ucimlrepo import fetch_ucirepo 
musk_version_2 = fetch_ucirepo(id=75)
```

Prepare train and test data


```python
from sklearn.model_selection import train_test_split
X = musk_version_2.data.features
y = musk_version_2.data.targets
feature_types = musk_version_2.variables.type
cols = X.columns
cat_cols = []

total_size = X.shape[0]
train_ratio = 0.7

X_train, X_test, y_train, y_test = train_test_split(X,y,
    train_size=train_ratio,
    shuffle=True)
```

### 1. Define the base model and cost-importance function

To use our feature selection algorithm one must define the `cost_importance_function`, a function that evaluates the objective on the given subset of features (validation cost of the model) and their importances.

The function takes the binary input vector (particle position), **trains the base model on the feature subset** represented by the particle position, **returns validation cost** of the model and the **importance vector**. The importance vector must be of the same length as position vector, i-th entry should be the importance of the i-th feature in the whole feature pool, unused features automatically get zero importance. The `cost_importance_function` is required to do the steps

In this example **we use 1 - recall as a cost function**, thus the algorithms tries to maximize the recall. We use LightGBM classifier as the base model and its importance scores of type `gain` (indicates how much a feature reduces the entropy). The validation strategy is 5-fold cross-validation and we choose recall as validation metric due to label 1 being rare ($\sim 15 \%$)

In principle any predictor can serve as a base model and any relevance metric may serve as importance score. For example, the latter can be a filter method (i.e. mutual information). Tree-based models are particularly suitable, because they have a built-in feature importance scores, that tell how much the model relies on each utilized feature in the current fit.


```python
from swarm_feature_selection import get_needed_importances
from sklearn.model_selection import cross_val_score

def validate_binary_feature_subset(
        x,
        train_data,
        cols,
        cat_cols=None,
        cv=5,
        n_jobs=1,
        cost_only=True):
    
    feature_subset = cols[np.nonzero(x)]
    if cat_cols is not None:
        cat_features = feature_subset[np.isin(feature_subset,cat_cols)]

    model = lgbm.LGBMClassifier(
        objective="binary",
        num_leaves=16,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=n_jobs,
        verbosity=-1,
        importance_type="gain"
        )
    
    X_train, y_train = train_data
    X_train = X_train[feature_subset]
    cost = 1-cross_val_score(model,X_train,y_train,
                           scoring="recall",cv=5,verbose=0).mean()
    if cost_only:
        return cost
    else:
        model.fit(X_train,y_train)
        return cost, model

def cost_importance_function(x):
    cost, model = validate_binary_feature_subset(
        x,
        train_data=(X_train,y_train.values.ravel()),
        cols=X_train.columns,
        cat_cols=cat_cols,
        cost_only=False
    )
    importance = get_needed_importances(
        full_set=cols,
        model=model)
    return cost, importance
```

### 2. Initialize particle positions and the optimizer
To generate the initial positions of the particles, we use agglomerative feature clustering. We split features into `n_clusters` groups by similarity (here absolute Spearman correlation) and initialize every position with a subset of size `n_clusters`, drawing one feature from every group/cluster.

Alternatively, we can initialize each position with a vector of i.i.d. Bernoulli variables. This way is simpler, yet it is much more likely to produce statistically redundant feature subsets


```python
from scipy import stats
from sklearn.cluster import FeatureAgglomeration
from swarm_feature_selection import get_clusters

distance_matrix = 1 - np.abs(stats.spearmanr(X_train)[0])
agglo = FeatureAgglomeration(n_clusters=50,
                             metric="precomputed",
                             linkage="complete")
agglo.fit(distance_matrix)
clusters = get_clusters(agglo.labels_)

init_position = np.vstack([np.isin(
    np.arange(X_train.shape[1]),
    [cluster[np.random.randint(len(cluster))] for cluster in clusters]
).astype(int) for k in range(16)
])

```

Having defined the distance matrix, velocity weights and initial positions, we can initialize the multi-swarm BPSO object. The velocity weights $w, c_1, c_2$ are hyperparameters inherited from vanilla PSO


```python
from swarm_feature_selection import RegLocalMultiBPSO
velocity_weights = {"w":0.5,"c1":0.5,"c2":0.5}
multi_bpso = RegLocalMultiBPSO(distance_matrix=distance_matrix,
                               velocity_weights=velocity_weights,
                               init_position=init_position,
                               n_subswarms=4)
```

### 3. Optimization loop

Now we run the optimization loop. At every iteration we need to run `evaluate_subswarms` and `step` methods. We also suggest to use `_regroup_subswarms` once in several iterations to enable more interactions in the multi-swarm. The regrouping mechanism is important for exploration and to avoid stagnation of the subswarms.

 The interface is low-level, so the user needs to manually call methods `evaluate_subswarms`, `step` and `_regroup_subswarms`. To run evaluation on multiple processes, one can define a pool of workers and provide it to `evaluate_subswarms`, or just specify `n_jobs` and the method will create a local pool, run evaluation and close it.

**Warning**: if `cost_importance_function` is defined locally and not in an importable module, the multiprocessing might not work: if you run the code below with `n_jobs>1`, the method `evaluate_subswarms` will probably get stuck and the following warning will be raised:\
 `StdErr from Kernel Process AttributeError: Can't get attribute 'cost_importance_function' on <module '__main__' (built-in)>`\
For the purpose of illustration, we defined `cost_importance_function` in the notebook, so we will only use 1 process.


```python
from functools import partial
import multiprocessing
from datetime import datetime

n_iters = 100
n_jobs = 1
regrouping_period = 10
pool = multiprocessing.Pool(n_jobs)
for epoch in range(n_iters):
    if epoch % regrouping_period == 0:
        multi_bpso._regroup_subswarms(random_state=epoch)
    start = datetime.now()
    print(f"Epoch {epoch+1} running...")
    multi_bpso.evaluate_subswarms(
        cost_importance_function=cost_importance_function,
        pool=pool,
        n_jobs=n_jobs
    )
    print([swarm.gbest_cost for swarm in multi_bpso.subswarms])
    multi_bpso.step(n_extra_searchers=1,local_search_prob=0.5)
    print(f"Finished in {(datetime.now()-start).total_seconds()} seconds")
pool.close()
```

    Epoch 1 running...
    [0.10610916860916864, 0.09490093240093245, 0.1033022533022534, 0.08233294483294495]
    Finished in 24.562553 seconds
    Epoch 2 running...
    [0.08516899766899777, 0.07115384615384612, 0.08378982128982138, 0.08233294483294495]
    Finished in 36.573689 seconds
    Epoch 3 running...
    [0.08516899766899777, 0.07115384615384612, 0.08378982128982138, 0.08233294483294495]
    Finished in 35.280752 seconds
    Epoch 4 running...
    [0.08515928515928517, 0.07115384615384612, 0.08378982128982138, 0.08233294483294495]
    Finished in 34.383487 seconds
    Epoch 5 running...
    [0.0767773892773892, 0.07115384615384612, 0.08378982128982138, 0.08233294483294495]
    Finished in 37.32088 seconds
    Epoch 6 running...
    [0.0767773892773892, 0.07115384615384612, 0.08378982128982138, 0.08233294483294495]
    Finished in 35.874723 seconds
    Epoch 7 running...
    [0.0767773892773892, 0.07115384615384612, 0.08378982128982138, 0.08233294483294495]
    Finished in 36.505518 seconds
    Epoch 8 running...
    [0.0767773892773892, 0.07115384615384612, 0.08378982128982138, 0.08233294483294495]
    Finished in 35.429117 seconds
    Epoch 9 running...
    ...
    Finished in 38.779093 seconds
    Epoch 100 running...
    [0.07116355866355872, 0.07114413364413363, 0.06840520590520582, 0.0642094017094017]
    Finished in 38.564606 seconds


The loop can be stopped and started again. One can also save the pbest or current positions of the subswarms and use them for initialization later.\
Below we stack the personal best positions of particles and the corresponding costs into one dataframe and save it as a CSV file.


```python
n_subswarms = 4
pbest_pos_cost = pd.DataFrame(
    data=np.vstack([multi_bpso.subswarms[k].pbest_pos for k in range(n_subswarms)]),
    columns=[f"pos{i}" for i in range(multi_bpso.subswarms[0].pbest_pos.shape[1])]
)
pbest_pos_cost["cost"] = np.concatenate(
    [multi_bpso.subswarms[k].pbest_cost
      for k in range(n_subswarms)]
)
# pbest_pos_cost.to_csv("pbest_pos_cost.csv")
```

### 4. Comparison of model performance on selected features vs full feature set
Since the multi-swarm optimization produces several feature subsets with improved validation error, it is best to ensemble the models trained using these subsets. This way the final model will be more robust and will likely have even smaller test error.


```python
from swarm_feature_selection import binary_to_subset

positions = pbest_pos_cost.drop(columns=["cost"]).values
best_index = np.argsort(pbest_pos_cost["cost"].values)
best_subsets = [binary_to_subset(x,full_set=cols)
                 for x in positions[best_index]]
```


```python
n_models = 8

y_preds = pd.DataFrame(index=y_test.index)
models = {f"clf_{i}":lgbm.LGBMClassifier(
        objective="binary",
        num_leaves=16,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=n_jobs,
        verbosity=-1,
        importance_type="gain"
        ) for i in range(n_models)}
for i in range(n_models):
    models[f"clf_{i}"].fit(
        X_train[best_subsets[i]],y_train.values.ravel()
    )
    y_preds[f"clf_{i}"] = models[f"clf_{i}"].predict(X_test[best_subsets[i]])
```

Below are classification quality metrics of the ensemble trained on selected features. We are interested in the recall of class 1.

```python
from sklearn.metrics import recall_score, classification_report
y_pred_agg = (y_preds.mean(axis=1)>0).astype(int)
print(classification_report(y_test,y_pred_agg,digits=3))
```

                  precision    recall  f1-score   support
    
             0.0      0.991     0.998     0.994      1679
             1.0      0.986     0.947     0.966       301
    
        accuracy                          0.990      1980
       macro avg      0.988     0.972     0.980      1980
    weighted avg      0.990     0.990     0.990      1980
    

We train a single baseline model without prior feature selection and display the same metrics.

```python
baseline_model = lgbm.LGBMClassifier(
        objective="binary",
        num_leaves=16,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=n_jobs,
        verbosity=-1,
        importance_type="gain"
        )
baseline_model.fit(X_train,y_train.values.ravel())
print(classification_report(y_test,baseline_model.predict(X_test),digits=3))
```

                  precision    recall  f1-score   support
    
             0.0      0.984     1.000     0.992      1679
             1.0      1.000     0.910     0.953       301
    
        accuracy                          0.986      1980
       macro avg      0.992     0.955     0.973      1980
    weighted avg      0.987     0.986     0.986      1980
    

The feature selection with ensembling improved the recall of positive class from 0.910 to 0.947.

## <a class="references" id="references">References</a>

* [M. Dorigo, M. Birattari and T. Stutzle, "Ant colony optimization," in IEEE Computational Intelligence Magazine, vol. 1, no. 4, pp. 28-39, Nov. 2006, doi: 10.1109/MCI.2006.329691.](https://ieeexplore.ieee.org/abstract/document/4129846?casa_token=JgeeCH5G2LcAAAAA:bDRukvnSA-DISDU3JQ5lqQCvrtKXTAl-Qq4nZ4qKNRIgn9QhaftbhjJfc_uEh5W20YWQ5k7mq8Hb)

* [Karaboga, D., Gorkemli, B., Ozturk, C., & Karaboga, N. (2014). A comprehensive survey: artificial bee colony (ABC) algorithm and applications. Artificial intelligence review, 42, 21-57.](https://link.springer.com/article/10.1007/s10462-012-9328-0)

* [J. J. Liang and P. N. Suganthan, "Dynamic multi-swarm particle swarm optimizer," Proceedings 2005 IEEE Swarm Intelligence Symposium, 2005. SIS 2005., Pasadena, CA, USA, 2005, pp. 124-129, doi: 10.1109/SIS.2005.1501611](https://ieeexplore.ieee.org/abstract/document/1501611?casa_token=X67JkuriAOsAAAAA:ajByfMa2Tsjil54yDXnbMbfFO9z8YOt1yeJKQ2IJqwdRIxbJ-Nbz0-OLyy8oL1ThH0PYmTecxW3f)

* [Yang, X. S. (2009, October). Firefly algorithms for multimodal optimization. In International symposium on stochastic algorithms (pp. 169-178). Berlin, Heidelberg: Springer Berlin Heidelberg.](https://arxiv.org/pdf/1003.1466)

