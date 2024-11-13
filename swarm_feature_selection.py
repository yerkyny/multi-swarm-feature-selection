import numpy as np
import scipy
import datetime
import math
from datetime import datetime, timedelta
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
import sklearn
import logging
import lightgbm as lgbm
import multiprocessing
from functools import partial
from copy import deepcopy
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import uniform

def get_clusters(a):
    """
    Returns:
        list of clusters, each cluster is a list of of indices belonging to the cluster
    """
    labels = np.unique(a)
    output = [list(np.where(a == label)[0]) for label in labels]
    return output

def used_feature_binary(
        full_set,
        model=None,
        model_feature_name=None,
        model_feature_importances=None,
        return_binary=True
):
    """ 
    Args:
        full_set (array_like): candidate columns set (pool)
        model (optional): fitted estimator. Must have attributes
          "feature_name_" and "feature_importances_". If None, model_feature_name 
          and model_feature_importances must be provided.
        model_feature_name (array_like, optional): array or list of feature names.
          Defaults to None
        model_feature_importances (array_like, optional): array or list of
          feature importances of the same shape as model_feature_name
          Defaults to None

    Returns:
        ndarray: binary mask for feature indices from full_set used by the model. 
        Must have the same shape as full_set
    """
    
    if model is not None:
        model_feature_name = model.feature_name_
        model_feature_importances = model.feature_importances_
    model_feature_name = np.array(model_feature_name)
    model_feature_importances = np.array(model_feature_importances)
    used_feature = model_feature_name[model_feature_importances>0]
    output = np.isin(full_set,used_feature).astype(int)
    return output

def binary_to_subset(x,full_set=None):
    if full_set is None:
        full_set = np.arange(len(x))
    return full_set[np.nonzero(x)]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def compute_cost(swarm_position, cost_function):
    if len(swarm_position.shape) == 1:
        return cost_function(swarm_position)
    else:
        output = np.zeros(swarm_position.shape[0])
        for i in range(swarm_position.shape[0]):
            output[i] = cost_function(swarm_position[i])
    return output

def apply_bounds(x, lower_bound, upper_bound):
    return np.minimum(np.maximum(x,lower_bound),upper_bound)

def get_needed_importances(full_set,model=None,model_feature_name=None,model_feature_importances=None):
    """ Returns feature importances of features in full_set in the order of feature names in full_set 
    of length len(full_set). Must be provided either the model or model's feature_name_ and feature_importances_ attributes.
    """
    if model is not None:
        model_feature_name = model.feature_name_
        model_feature_importances = model.feature_importances_
    used_importances = np.zeros(full_set.shape)
    tmp = pd.DataFrame(data=model_feature_importances.reshape(1,-1),
                       columns=model_feature_name)
    name_mask = np.isin(full_set,model_feature_name) # mask of pool features in feature_name
    used_importances[name_mask] = tmp[full_set[name_mask]].values.ravel() # importances of pool features
    return used_importances


def compute_local_search_position(
        importances,
        distance_matrix):
    n_used_feature = np.sum(importances>0)
    # removal step
    eta = np.random.randint(n_used_feature) # number of candidates to remove
    thresh = np.sort(importances[importances>0])[eta] 
    removal_mask = (importances>0)*(importances<=thresh) # removal candidates mask
    used_mask = importances>0 # used pool features
    output = used_mask.astype(int)
    output[removal_mask] = (uniform.rvs(size=removal_mask.sum())>0.5).astype(int) # removal step done
    
    # addition step
    distances = np.sum(distance_matrix[used_mask,:]**2,axis=0)
    distances_unused = np.sum(distance_matrix[used_mask,:][:,~used_mask]**2,axis=0)
    theta = np.random.randint(min(n_used_feature,
                                  len(importances)-n_used_feature)) # number of candidates to add
    dist_thresh = np.sort(distances_unused)[::-1][theta]
    add = (distances * ~used_mask >= dist_thresh)
    add = add * (uniform.rvs(size=importances.shape)>0.5).astype(int)
    output = output + add
    return output

class Swarm():
    
    def __init__(self,
            position=None,
            velocity=None,
            from_history=False,
            history=None):
        
        if from_history==False:
            self.position = position
            self.velocity = velocity
            self.n_particles = position.shape[0]
            self.dimensions = position.shape[1]
            self.pbest_pos = deepcopy(position)
            self.gbest_pos = deepcopy(position[0])
            self.pbest_cost = np.ones(self.n_particles) * np.inf
            self.gbest_cost = np.inf
            self.current_cost = np.ones(self.n_particles) * np.inf
            self.status = "Requires evaluation"
            self.history = []
            self.pbest_importances = np.zeros_like(self.position)
        else:
            self.history = deepcopy(history)
            self.position = self.history[-1]["position"]#.astype(int)
            self.velocity = self.history[-1]["velocity"]
            self.n_particles = self.position.shape[0]
            self.dimensions = self.position.shape[1]
            self.pbest_pos = self.history[-1]["pbest_pos"]#.astype(int)
            self.gbest_pos = self.history[-1]["gbest_pos"]#.astype(int)
            self.pbest_cost = self.history[-1]["pbest_cost"]
            self.gbest_cost = self.history[-1]["gbest_cost"]
            self.current_cost = self.history[-1]["current_cost"]
            if "pbest_importances" in history.keys():
                self.pbest_importances = history["pbest_importances"]
            else:
                self.pbest_importances = np.zeros_like(self.position)
        
    
    def set_status(self,status):
        self.status = status

    def update_best(self, current_importances=None):
        for i in range(self.n_particles):
            if self.current_cost[i] < self.pbest_cost[i]:
                self.pbest_cost[i] = self.current_cost[i]
                self.pbest_pos[i] = self.position[i]
                if current_importances is not None:
                    self.pbest_importances[i] = current_importances[i]
                    self.pbest_pos[i] = (current_importances[i]>0).astype(int)
        self.gbest_cost = np.min(self.pbest_cost)
        self.gbest_pos = self.pbest_pos[np.argmin(self.pbest_cost)]
        self.status = "Ready for step"

    def update_history(self):
        new_record = {
            "position":self.position,
            "velocity":self.velocity,
            "current_cost":self.current_cost,
            "pbest_pos":self.pbest_pos,
            "gbest_pos":self.gbest_pos,
            "pbest_cost":self.pbest_cost,
            "gbest_cost":self.gbest_cost,
            "pbest_importances":self.pbest_importances,
            "status":self.status
        }
        self.history.append(new_record)


class BinaryParticleSwarmOptimization():

    def __init__(self,
                n_particles=None,
                dimensions=None,
                init_position=None,
                init_velocity=None,
                init_swarm=None,
                velocity_weights=None,
                ) -> None:
        """Initialize binary swarm population

        Args:
            n_particles (int): number of particles, i.e. size of the swarm
            dimensions (int): dimensionality of search space
            init_position (ndarray, optional): initial position of the swarm, must be a numpy array of shape (n_particles,dimensions).
                Defaults to None.
            init_velocity (ndarray, optional): initial velocity of the swarm, must be a numpy array of shape (n_particles,dimensions).
                Defaults to None.
            init_swarm (Swarm object, optional): initial swarm. If not None, init_position and init_velocity will be ignored.
                Defaults to None.
            velocity_weights (dict, optional): weights "w", "c1" and "c2" for the velocity update.
                Defaults to None.
        """
        if init_swarm is None:
            self.n_particles = n_particles
            self.dimensions = dimensions
            if init_position is not None:
                self.init_position = init_position
            else:
                self.init_position = self._init_random_position(
                    n_particles=n_particles,
                    dimensions=dimensions
                    )
            if init_velocity is not None:
                self.init_velocity = init_velocity
            else:
                self.init_velocity = apply_bounds(
                    self._init_random_velocity(
                    n_particles=n_particles,
                    dimensions=dimensions),
                -2,2)
            self.swarm = Swarm(self.init_position, self.init_velocity)
        else:
            self.swarm = init_swarm
        if velocity_weights is not None:
            self.velocity_weights = velocity_weights
        else:
            self.velocity_weights = {"w":1,"c1":1,"c2":1}
        
    
    def evaluate_swarm(self,
                        cost_function,
                        n_jobs=None,
                        verbose=False):
        """ Computes the cost function at the current swarm positions and updates gbest and pbest.

        Args:
            cost_function (function): accepts binary vector and returns a scalar
            n_jobs (int, optional): number of concurrent workers. Defaults to None.
        """
        # compute cost function
        
        if n_jobs in [None,1]:
            self.swarm.current_cost = compute_cost(swarm_position=self.swarm.position,
                                                    cost_function=cost_function)
        else:
            pool = multiprocessing.Pool(n_jobs)
            results = pool.map(
                partial(compute_cost, cost_function=cost_function),
                np.array_split(self.swarm.position, n_jobs)
            )
            pool.close()
            self.swarm.current_cost = np.concatenate(results)
        self.swarm.update_best()

    def update_swarm_history(self):
        self.swarm.update_history()

    def step(self):
        next_velocity = self._compute_next_velocity(self.swarm)
        self.swarm.velocity = next_velocity
        next_position = self._compute_next_position(self.swarm)
        self.swarm.position = next_position
        self.swarm.status = "Requires evaluation"

    def _compute_next_velocity(self, swarm):
        swarm_size=swarm.velocity.shape
        velocity = (
            self.velocity_weights["w"] * swarm.velocity+
            self.velocity_weights["c1"] * uniform.rvs(size=swarm_size)
            * (swarm.pbest_pos - swarm.position) +
            self.velocity_weights["c2"] * uniform.rvs(size=swarm_size)
            * (swarm.gbest_pos - swarm.position)
        )
        return apply_bounds(np.copy(velocity),-2,2)

    def _init_random_position(self,n_particles,dimensions,freq=0.5):
        return (uniform.rvs(size=(n_particles,dimensions))>freq).astype(int)

    def _init_random_velocity(self,n_particles,dimensions,std=1):
        return scipy.stats.norm.rvs(0,std,size=(n_particles,dimensions))

    def _compute_next_position(self, swarm):
        return (
                uniform.rvs(size=swarm.velocity.shape)<
                sigmoid(swarm.velocity)
            )
    
class LocalSearchBPSO(BinaryParticleSwarmOptimization):
    
    def __init__(self,
                distance_matrix,
                n_particles=None,
                dimensions=None,
                init_position=None,
                init_velocity=None,
                init_swarm=None,
                velocity_weights=None,
                ) -> None:
        
        super().__init__(n_particles,
                dimensions,
                init_position,
                init_velocity,
                init_swarm,
                velocity_weights)
        self.distance_matrix = distance_matrix

    def evaluate_swarm(self, cost_importance_function, pool=None, n_jobs = None,verbose=False):
        """ Computes cost function at the curent swarm position, while keeping track 
        of features with non-zero feature importances

        Args:
            cost_function (_type_): must return both the 
            n_jobs (int, optional): _description_. Defaults to None.
            pool ()
            verbose (bool, optional): _description_. Defaults to False.
        """
        if n_jobs in [None,1] and pool is None:
            self.swarm.current_cost, importances = self._compute_cost_and_importance(
                swarm_position=self.swarm.position,
                cost_importance_function=cost_importance_function
            )
        elif pool is None:
            pool = multiprocessing.Pool(n_jobs)
            results = pool.map(
                partial(self._compute_cost_and_importance,
                         cost_importance_function=cost_importance_function),
                np.array_split(self.swarm.position, n_jobs)
            ) # 
            pool.close()
            self.swarm.current_cost = np.concatenate([res[0] for res in results])
            importances = np.vstack([res[1] for res in results])
        else:
            results = pool.map(
                partial(self._compute_cost_and_importance,
                         cost_importance_function=cost_importance_function),
                np.array_split(self.swarm.position, n_jobs)
            ) # 
            self.swarm.current_cost = np.concatenate([res[0] for res in results])
            importances = np.vstack([res[1] for res in results])

        self.swarm.update_best(current_importances=importances)

    def step(self,n_extra_searchers=0,local_search_prob=0.5):
        next_velocity = self._compute_next_velocity(self.swarm)
        self.swarm.velocity = next_velocity
        next_position = self._compute_next_position(self.swarm,n_extra_searchers,local_search_prob)
        self.swarm.position = next_position
        self.swarm.status = "Requires evaluation"

    def _compute_cost_and_importance(self, swarm_position, cost_importance_function):
        """ Returns a tuple (costs, importances), costs contains cost values for every particle in swarm_position,
        importances is a vertical stack of importances of all particles in swarm_position

        Args:
            swarm_position (ndarray): array of particle positions, possibly 2d
            cost_impotance_function (function): must take a single particle position and return its cost
              and importances of length dimensions.
        """
        if len(swarm_position.shape)==1:
            return cost_importance_function(swarm_position)
        else:
            costs = np.zeros(swarm_position.shape[0])
            importances = np.zeros(swarm_position.shape)
            for i in range(swarm_position.shape[0]):
                costs[i], importances[i] = cost_importance_function(swarm_position[i])
        return costs, importances
    
    def _compute_next_velocity(self, swarm):
        return super()._compute_next_velocity(swarm)
    
    def _compute_next_position(self, swarm, n_extra_searchers=0, local_search_prob=0.5):
        """ The holder of gbest_cost alsways performs local search. The n_extra_searchers holding the best pbest_cost
        except gbest will perform local search with probability local_search_prob. 

        Args:
            swarm (Swarm object): swarm
            n_extra_searcher (int, optional): number of extra searchers. Defaults to 0.
            local_search_prob (float, optional): probability of local search for extra searchers. Defaults to 0.5.
        """
        output = (
            uniform.rvs(size=swarm.velocity.shape)<sigmoid(swarm.velocity)
        ).astype(int) # default position update
        if np.allclose(swarm.pbest_importances, np.zeros_like(swarm.position)):
            return output
        # compute local search position for gbest
        best_index = np.argsort(swarm.gbest_cost)
        output[best_index[0]] = compute_local_search_position(
            swarm.pbest_importances[best_index[0]], self.distance_matrix)
        # compute local search position for the extra searchers
        for i in best_index[1:n_extra_searchers+1]:
            if uniform.rvs(size=1)<local_search_prob:
                output[i] = compute_local_search_position(
                    swarm.pbest_importances[i],self.distance_matrix)
        return output
    
    def _init_random_position(self,n_particles,dimensions,freq=0.5):
        return (uniform.rvs(size=(n_particles,dimensions))>freq).astype(int)

    def _init_random_velocity(self,n_particles,dimensions,std=1):
        return scipy.stats.norm.rvs(0,std,size=(n_particles,dimensions))



class RegLocalMultiBPSO(LocalSearchBPSO):

    def __init__(self,
            distance_matrix,
            velocity_weights,
            init_position=None,
            init_velocity=None,
            init_subswarm_indices=None,
            n_subswarms = None,
            init_subswarms=None
            ) -> None:
        
        self.distance_matrix = distance_matrix
        self.velocity_weights = velocity_weights
        if init_subswarms is None:
            self.init_position = init_position
            self.n_particles = init_position.shape[0]
            self.dimensions = init_position.shape[1]
            if init_velocity is not None:
                self.init_velocity = init_velocity
            else:
                self.init_velocity = apply_bounds(
                    self._init_random_velocity(
                    n_particles=self.n_particles,
                    dimensions=self.dimensions),
                -2,2)
            if init_subswarm_indices is None:
                self.n_subswarms = n_subswarms
                init_subswarm_indices = np.array(np.array_split(range(self.n_particles),n_subswarms))
            self.subswarms = [Swarm(self.init_position[init_subswarm_indices[k]],
                                    self.init_velocity[init_subswarm_indices[k]])
                                    for k in range(self.n_subswarms)]
        else:
            self.subswarms = deepcopy(init_subswarms) # list of Swarm objects
            self.n_subswarms = len(init_subswarms)
            self.dimensions = self.subswarms[0].dimensions
            self.n_particles = sum(swarm.n_particles for swarm in self.subswarms)
        
        self.subswarm_indices = np.split(
            np.arange(self.n_particles),
            np.cumsum([swarm.n_particles for swarm in self.subswarms])
            )[:self.n_subswarms]
        
    def evaluate_subswarms(self, cost_importance_function, pool=None, n_jobs=None):
        
        subswarm_position_stack = np.vstack(
            [self.subswarms[k].position for k in range(self.n_subswarms)])
        if n_jobs in [None,1]:
            current_cost_stack, importances_stack = self._compute_cost_and_importance(
                swarm_position=subswarm_position_stack,
                cost_importance_function=cost_importance_function
            )
        elif pool is None:
            pool = multiprocessing.Pool(n_jobs)
            results = pool.map(
                partial(self._compute_cost_and_importance,
                         cost_importance_function=cost_importance_function),
                np.array_split(subswarm_position_stack, n_jobs)
            )
            pool.close()
            current_cost_stack = np.concatenate([res[0] for res in results])
            importances_stack = np.vstack([res[1] for res in results])
        else:
            results = pool.map(
                partial(self._compute_cost_and_importance,
                         cost_importance_function=cost_importance_function),
                np.array_split(subswarm_position_stack, n_jobs)
            )
            current_cost_stack = np.concatenate([res[0] for res in results])
            importances_stack = np.vstack([res[1] for res in results])

        for k in range(self.n_subswarms):
            self.subswarms[k].current_cost = current_cost_stack[self.subswarm_indices[k]]
            self.subswarms[k].update_best(current_importances=importances_stack[self.subswarm_indices[k]])
            

    def _compute_cost_and_importance(self, swarm_position, cost_importance_function):
        """ Returns a tuple (costs, importances), costs contains cost values for every particle in swarm_position,
        importances is a vertical stack of importances of all particles in swarm_position

        Args:
            swarm_position (ndarray): array of particle positions, possibly 2d
            cost_impotance_function (function): must take a single particle position and return its cost
              and importances of length dimensions.
        """
        if len(swarm_position.shape)==1:
            return cost_importance_function(swarm_position)
        else:
            costs = np.zeros(swarm_position.shape[0])
            importances = np.zeros(swarm_position.shape)
            for i in range(swarm_position.shape[0]):
                costs[i], importances[i] = cost_importance_function(swarm_position[i])
        return costs, importances
    
    def step(self, n_extra_searchers=0, local_search_prob=0.5):
        for k in range(self.n_subswarms):
            next_velocity = self._compute_next_velocity(self.subswarms[k])
            self.subswarms[k].velocity = next_velocity
            next_position = self._compute_next_position(self.subswarms[k],n_extra_searchers,local_search_prob)
            self.subswarms[k].position = next_position
            self.subswarms[k].status = "Requires evaluation"
    
    def _regroup_subswarms(self, random_state=None):

        np.random.seed(random_state)
        permuted_index = np.random.permutation(self.n_particles)
        status = self.subswarms[0].status
        subswarm_position_stack = np.vstack(
            [self.subswarms[k].position for k in range(self.n_subswarms)])
        subswarm_velocity_stack = np.vstack(
            [self.subswarms[k].velocity for k in range(self.n_subswarms)]) # maybe we will reinitialize velocities after regrouping
        pbest_pos_stack = np.vstack(
            [self.subswarms[k].pbest_pos for k in range(self.n_subswarms)])
        pbest_cost_stack = np.concatenate(
            [self.subswarms[k].pbest_cost for k in range(self.n_subswarms)])
        pbest_importances_stack = np.vstack(
            [self.subswarms[k].pbest_importances for k in range(self.n_subswarms)])
        current_cost_stack = np.concatenate(
            [self.subswarms[k].current_cost for k in range(self.n_subswarms)])
        self.subswarms = [Swarm(
            position=subswarm_position_stack[permuted_index[self.subswarm_indices[k]]],
            velocity=subswarm_velocity_stack[permuted_index[self.subswarm_indices[k]]]
        ) for k in range(self.n_subswarms)]

        for k in range(self.n_subswarms):
            self.subswarms[k].pbest_pos = pbest_pos_stack[permuted_index[self.subswarm_indices[k]]]
            self.subswarms[k].pbest_cost = pbest_cost_stack[permuted_index[self.subswarm_indices[k]]]
            self.subswarms[k].pbest_importances = pbest_importances_stack[permuted_index[self.subswarm_indices[k]]]
            self.subswarms[k].current_cost = current_cost_stack[permuted_index[self.subswarm_indices[k]]]
            self.subswarms[k].gbest_cost = np.min(self.subswarms[k].pbest_cost)
            self.subswarms[k].gbest_pos = self.subswarms[k].pbest_pos[np.argmin(self.subswarms[k].pbest_cost)]
            self.subswarms[k].status = status

    def _compute_next_velocity(self, swarm):
        return super()._compute_next_velocity(swarm)
    
    def _compute_next_position(self, swarm, n_extra_searchers=0, local_search_prob=0.5):
        return super()._compute_next_position(swarm, n_extra_searchers, local_search_prob)

    def _init_random_velocity(self,n_particles,dimensions,std=1):
        return scipy.stats.norm.rvs(0,std,size=(n_particles,dimensions))
