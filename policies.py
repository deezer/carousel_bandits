from collections import defaultdict
from math import log
from online_logistic_regression import OnlineLogisticRegression
from scipy.special import expit
from scipy.optimize import minimize
import numpy as np


# SOME COMMENTS ON THE IMPLEMENTATION OF A "CASCADING BROWSING BEHAVIOUR"
#
# Implementing actual cascade-based browsing user behaviours can be tricky.
# Throughout these offline simulations, we made the following implementation choices:
#
# - Users browse the carousel from left to right. If a user clicked and streamed a playlist
# at rank r with 1 <= r <= n_recos, he/she stops browsing and all playlists from ranks
# r+1 to n_recos are considered as "unseen" and corresponding arms are not updated.
#
# - Moreover, as users can be drawn several times in the batch of a same round, a same user can
# have several "browsing sessions" during a same round. Therefore, each user can have several positive
# rewards - i.e. stream several playlists - in a same round, consistently with our multiple-plays framework.
#
# - In the below classes, we shuffle the first L_init playlists, capturing the fact that they are all
# initially visible on the user's screen (and thus the aforementioned "left-to-right" browsing
# behaviour is not relevant). Note: overall, adding this shuffle does not impact performances.
#
# - We consider that a user (from a selected batch) that did not stream any playlist (all rewards = 0)
# only saw the L_init first ones. Arms of playlists positioned further in the carousel are not updated.


# Abstract class defining the minimal functions that need
# to be implemented to create new bandit policy classes
class Policy:

    # Returns a list of size n_recos of playlist ids
    def recommend_to_users_batch(self, batch_users, n_recos=12):
        return

    # Updates policies parameters
    def update_policy(self, user_ids, recos , rewards):
        return


# A simple baseline that randomly recommends n_recos playlists to each user.
class RandomPolicy(Policy):
    def __init__(self, n_playlists, cascade_model=True):
        self.cascade_model = cascade_model
        self.n_playlists = n_playlists

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        n_users = len(batch_users)
        recos = np.zeros((n_users, n_recos), dtype=np.int64)
        r = np.arange(self.n_playlists)
        for i in range(n_users):
            np.random.shuffle(r)
            recos[i] = r[:n_recos]
        return recos

    def update_policy(self, user_ids, recos, rewards, l_init=3):
        return


#  Upper Confidence Bound (UCB) strategy, using KL-UCB bounds [Garivier and Cappe, 2011] tailored for Bernoulli rewards
class KLUCBSegmentPolicy(Policy):
    def __init__(self, user_segment, n_playlists, precision = 1e-6, eps = 1e-15, cascade_model=True):
        self.user_segment = user_segment
        n_segments = len(np.unique(self.user_segment))
        self.playlist_display = np.zeros((n_segments, n_playlists))
        self.playlist_success = np.zeros((n_segments, n_playlists))
        self.playlist_score = np.ones((n_segments, n_playlists))
        self.t = 0
        self.cascade_model = cascade_model
        self.precision = precision
        self.eps = eps

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        user_segment = np.take(self.user_segment, batch_users)
        user_score = np.take(self.playlist_score, user_segment, axis = 0)
        # Break ties
        user_random_score = np.random.random(user_score.shape)
        user_choice = np.lexsort((user_random_score, -user_score))[:, :n_recos]
        # Shuffle l_init first slots
        np.random.shuffle(user_choice[0:l_init])
        return user_choice

    def kl(self, x, y):
        x = min(max(x, self.eps), 1 - self.eps)
        y = min(max(y, self.eps), 1 - self.eps)
        return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))

    def scoring_function(self, n_success, n, t):
        if n == 0:
            return 1.0
        p = n_success / n
        value = p
        u = 1
        threshold = log(t) / n
        _count_iteration = 0
        while _count_iteration < 50 and u - value > self.precision:
            _count_iteration += 1
            m = (value + u) * 0.5
            if self.kl(p, m) > threshold:
                u = m
            else:
                value = m
        return (value + u) * 0.5

    def update_policy(self, user_ids, recos, rewards, l_init=3):
        batch_size = len(user_ids)
        modified_data = defaultdict(set)
        for i in range(batch_size):
            user_segment = self.user_segment[user_ids[i]]
            total_stream = len(rewards[i].nonzero())
            nb_display = 0
            for p, r in zip(recos[i], rewards[i]):
                nb_display +=1
                modified_data[user_segment].add(p)
                self.playlist_success[user_segment][p]+=r
                self.playlist_display[user_segment][p]+=1
                if self.cascade_model and ((total_stream == 0 and nb_display == l_init) or (r == 1)):
                    break
        self.t = self.playlist_display.sum()
        for seg,pls in modified_data.items():
            for pl in pls:
                self.playlist_score[seg][pl] = self.scoring_function(self.playlist_success[seg][pl], self.playlist_display[seg][pl], self.t)
        return


# An Explore-then-Commit strategy: similar to random until each playlist has been displayed n times or more,
# then recommends the top n_reco playlists with highest mean observed rewards, for each segment
class ExploreThenCommitSegmentPolicy(Policy):
    def __init__(self, user_segment, n_playlists, min_n, cascade_model=True):
        self.user_segment = user_segment
        n_segments = len(np.unique(self.user_segment))
        self.playlist_display = np.zeros((n_segments, n_playlists))
        self.playlist_success = np.zeros((n_segments, n_playlists))
        self.min_n = min_n
        self.cascade_model = cascade_model

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        user_segment = np.take(self.user_segment, batch_users)
        user_success = np.take(self.playlist_success, user_segment, axis = 0)
        user_displays = np.take(self.playlist_display, user_segment, axis = 0).astype(float)
        user_random_score = np.random.random(user_displays.shape)
        user_score = np.divide(user_success, user_displays, out=np.zeros_like(user_displays), where=user_displays!=0)
        discounted_displays = np.maximum(np.zeros_like(user_displays), self.min_n - user_displays)
        user_choice = np.lexsort((user_random_score, - user_score, -discounted_displays))[:, :n_recos]
        # Shuffle l_init first slots
        np.random.shuffle(user_choice[0:l_init])
        return user_choice

    def update_policy(self, user_ids, recos , rewards, l_init=3):
        batch_size = len(user_ids)
        for i in range(batch_size):
            user_segment = self.user_segment[user_ids[i]]
            total_stream = len(rewards[i].nonzero())
            nb_display = 0
            for p, r in zip(recos[i], rewards[i]):
                nb_display +=1
                self.playlist_success[user_segment][p]+=r
                self.playlist_display[user_segment][p]+=1
                if self.cascade_model and ((total_stream == 0 and nb_display == l_init) or (r == 1)):
                    break
        return


# Segment-based Epsilon-Greedy strategy: recommends playlists randomly with probability epsilon,
# otherwise recommends the top n_recos with highest mean observed rewards.
class EpsilonGreedySegmentPolicy(Policy):
    def __init__(self, user_segment, n_playlists, epsilon, cascade_model=True):
        self.user_segment = user_segment
        n_segments = len(np.unique(self.user_segment))
        self.playlist_display = np.zeros((n_segments, n_playlists))
        self.playlist_success = np.zeros((n_segments, n_playlists))
        self.playlist_score = np.ones((n_segments, n_playlists))
        self.epsilon = epsilon
        self.cascade_model = cascade_model

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        user_segment = np.take(self.user_segment, batch_users)
        user_scores = np.take(self.playlist_score, user_segment, axis = 0)
        user_random_score = np.random.random(user_scores.shape)
        n_users = len(batch_users)
        user_greedy = np.random.binomial(1, [1- self.epsilon for i in range(n_users)])
        new_scores = user_scores * user_greedy[:,np.newaxis]
        user_choice = np.lexsort((user_random_score, -new_scores))[:, :n_recos]
        # Shuffle l_init first slots
        np.random.shuffle(user_choice[0:l_init])
        return user_choice

    def update_policy(self, user_ids, recos, rewards, l_init=3):
        batch_size = len(user_ids)
        for i in range(batch_size):
            user_segment = self.user_segment[user_ids[i]]
            total_stream = len(rewards[i].nonzero())
            nb_display = 0
            for p, r in zip(recos[i], rewards[i]):
                nb_display +=1
                self.playlist_success[user_segment][p]+=r
                self.playlist_display[user_segment][p]+=1
                self.playlist_score[user_segment][p] = self.playlist_success[user_segment][p] / self.playlist_display[user_segment][p]
                if self.cascade_model and ((total_stream == 0 and nb_display == l_init) or (r == 1)):
                    break
        return


# Segment-based Thompson Sampling strategy, with Beta(alpha_zero,beta_zero) priors
class TSSegmentPolicy(Policy):
    def __init__(self, user_segment, n_playlists, alpha_zero=1, beta_zero=99, cascade_model=True):
        self.user_segment = user_segment
        n_segments = len(np.unique(self.user_segment))
        self.playlist_display = np.zeros((n_segments, n_playlists))
        self.playlist_success = np.zeros((n_segments, n_playlists))
        self.alpha_zero = alpha_zero
        self.beta_zero = beta_zero
        self.t = 0
        self.cascade_model = cascade_model

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        user_segment = np.take(self.user_segment, batch_users)
        user_displays = np.take(self.playlist_display, user_segment, axis = 0).astype(float)
        user_success = np.take(self.playlist_success, user_segment, axis = 0)
        user_score = np.random.beta(self.alpha_zero+user_success, self.beta_zero+user_displays - user_success)
        user_choice = np.argsort(-user_score)[:, :n_recos]
        # Shuffle l_init first slots
        np.random.shuffle(user_choice[0:l_init])
        return user_choice

    def update_policy(self, user_ids, recos , rewards, l_init = 3):
        batch_size = len(user_ids)
        for i in range(batch_size):
            user_segment = self.user_segment[user_ids[i]]
            total_stream = len(rewards[i].nonzero())
            nb_display = 0
            for p, r in zip(recos[i], rewards[i]):
                nb_display +=1
                self.playlist_success[user_segment][p]+=r
                self.playlist_display[user_segment][p]+=1
                if self.cascade_model and ((total_stream == 0 and nb_display == l_init) or (r == 1)):
                    break
        return


# Linear Thompson Sampling strategy for fully personalized contextual bandits, as in [Chapelle and Li, 2011]
class LinearTSPolicy(Policy):
    def __init__(self, user_features, n_playlists, bias=0.0, cascade_model=True):
        self.user_features = user_features
        n_dim = user_features.shape[1]
        self.n_playlists = n_playlists
        self.models = [OnlineLogisticRegression(1, 1, n_dim, bias, 15) for i in range(n_playlists)]
        self.m = np.zeros((n_playlists, n_dim))
        self.m[:, -1] = bias
        self.q = np.ones((n_playlists, n_dim))
        self.n_dim = n_dim
        self.cascade_model = cascade_model

    def recommend_to_users_batch(self, batch_users, n_recos=12, l_init=3):
        user_features = np.take(self.user_features, batch_users, axis=0)
        n_users = len(batch_users)
        recos = np.zeros((n_users, n_recos), dtype=np.int64)
        step = 1
        u = 0
        while u < n_users:
            u_next = min(n_users, u+step)
            p_features_sampled =(np.random.normal(self.m, 1/np.sqrt(self.q), size= (u_next-u, self.n_playlists, self.n_dim)))
            step_p = p_features_sampled.dot(user_features[u:u_next].T)
            for i in range(u_next - u):
                recos[u+i] = np.argsort((-step_p[i,:,i]))[:n_recos]
            u += step
        # Shuffle l_init first slots
        np.random.shuffle(recos[0:l_init])
        return recos

    def update_policy(self, user_ids, recos , rewards, l_init=3):
        rewards = 2*rewards - 1
        batch_size = len(user_ids)
        modified_playlists ={}
        for i in range(batch_size):
            total_stream = len(rewards[i].nonzero())
            nb_display = 0
            for p, r in zip(recos[i], rewards[i]):
                nb_display +=1
                if p not in modified_playlists:
                    modified_playlists[p] = {"X" : [], "Y" : []}
                modified_playlists[p]["X"].append(self.user_features[user_ids[i]])
                modified_playlists[p]["Y"].append(r)
                if self.cascade_model and ((total_stream == 0 and nb_display == l_init) or (r == 1)):
                    break
        for p,v in modified_playlists.items():
            X = np.array(v["X"])
            Y = np.array(v["Y"])
            self.models[p].fit(X,Y)
            self.m[p] = self.models[p].m
            self.q[p] = self.models[p].q
        return