"""
The code for 'class GaussianMixture()' is modified based on https://github.com/ldeecke/gmm-torch
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from utils import calculate_matmul, calculate_matmul_n_times
import pdb


class GaussianMixture():
    def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-4, init_params="random", mu_init=None, var_init=None):
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = self.mu_init.copy()
        else:
            self.mu = np.random.randn(1, self.n_components, self.n_features)

        if self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)  --> (1,k,d)
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
                self.var = self.var_init.copy()
            else:
                self.var = np.ones(self.n_features).reshape(1, 1, self.n_features).repeat(self.n_components,1)

        # (1, k, 1)
        self.wts = np.ones((1, self.n_components, 1), float)*1.0/self.n_components
        self.params_fitted = False

    def check_size(self, x):
        if len(np.shape(x)) == 2:
            # (n, d) --> (n, 1, d)
            x = x[:, np.newaxis, :]
        return x

    def fit(self, x, delta=1e-4, n_iter=500):
        if self.params_fitted:   # mu:[1,2,1],  var:[1,2,1,1], wts:[1,2,1]
            self._init_params()   # -->mu:[1,2,5], var:[1,2,5], wts:[1,2,1]

        x = self.check_size(x)    # x:[np,1,1]  --> [np,1,5]

        i = 0
        j = np.inf
        min_n_iter = 30

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu.copy()
            var_old = self.var.copy()

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if np.isinf(np.abs(self.log_likelihood)) or np.isnan(self.log_likelihood):
                self.__init__(self.n_components,
                              self.n_features,
                              covariance_type=self.covariance_type,
                              mu_init=self.mu_init,
                              var_init=self.var_init,
                              eps=self.eps)

                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

                if i <= min_n_iter:
                    i = 0
                    j = np.inf
                    self.__init__(self.n_components,
                                  self.n_features,
                                  covariance_type=self.covariance_type,
                                  mu_init=self.mu_init,
                                  var_init=self.var_init,
                                  eps=self.eps)

        self.params_fitted = True

    def predict(self, x, probs=True):
        x = self.check_size(x)
        weighted_log_prob = self._estimate_log_prob(x[:,:,0:1], id_=0) + \
                            self._estimate_log_prob(x[:, :, 1:2], id_=1) + \
                            self._estimate_log_prob(x[:, :, 2:3], id_=2) + \
                            self._estimate_log_prob(x[:, :, 3:4], id_=3) + \
                            self._estimate_log_prob(x[:, :, 4:5], id_=4) + \
                            np.log(self.wts)   # [np,2,1]

        if probs:
            p_k = np.exp(weighted_log_prob)
            return np.squeeze(p_k / (p_k.sum(1, keepdims=True)))
        else:
            return np.squeeze(np.argmax(weighted_log_prob, 1))

    def predict_prob(self, x):
        return self.predict(x, probs=True)

    def _estimate_log_prob(self, x, id_):
        x = self.check_size(x)
        if self.covariance_type == "full":
            mu = self.mu[:,:,id_].copy()[:,:,np.newaxis]
            var = self.var[:,:,id_].copy()[:,:,np.newaxis,np.newaxis]

            precision = np.linalg.inv(var)
            log_2pi = x.shape[-1] * np.log(2.*np.pi)
            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu)[:,:,np.newaxis,:]
            x_mu = (x - mu)[:,:,:,np.newaxis]
            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)


    def _calculate_log_det(self, var):
        log_det = np.empty((self.n_components,))

        for k in range(self.n_components):
            log_det[k] = 2 * np.log(np.diagonal(np.linalg.cholesky(var[0, k]))).sum()

        return log_det[:, np.newaxis]

    def _e_step(self, x):
        x = self.check_size(x)  # [np,1,1]
        weighted_log_prob = self._estimate_log_prob(x[:,:,0:1], id_=0) + \
                            self._estimate_log_prob(x[:, :, 1:2], id_=1) + \
                            self._estimate_log_prob(x[:, :, 2:3], id_=2) + \
                            self._estimate_log_prob(x[:, :, 3:4], id_=3) + \
                            self._estimate_log_prob(x[:, :, 4:5], id_=4) + \
                            np.log(self.wts)   # [np,2,1]

        log_prob_norm = logsumexp(weighted_log_prob, axis=1, keepdims=True)  # [np,1,1]
        log_resp = weighted_log_prob - log_prob_norm   # [np,2,1]

        return np.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        x = self.check_size(x)  # [np,1,1]
        resp = np.exp(log_resp)   # [np,2,1]
        wts = np.sum(resp, axis=0, keepdims=True) + self.eps   # [1,2,1]
        mu = np.sum(resp*x, axis=0, keepdims=True) / wts  # [1,2,1]

        eps = np.eye(1) * self.eps    # [1,1]
        var = np.zeros((1,self.n_components,5), float)
        for fi in range(5):
            temp = np.sum(np.matmul((x[:,:,fi:fi+1] - mu[:,:,fi:fi+1])[:,:,:,np.newaxis], ((x[:,:,fi:fi+1] - mu[:,:,fi:fi+1])[:,:,np.newaxis,:])) * resp[:,:,:,np.newaxis], axis=0, keepdims=True) / np.sum(resp, axis=0, keepdims=True)[:,:,:,np.newaxis] + eps
            # [1,2,1,1]
            var[:,:,fi] = temp.squeeze()

        wts = wts / x.shape[0]

        return wts, mu, var

    def __em(self, x):
        _, log_resp = self._e_step(x)
        wts, mu, var = self._m_step(x, log_resp)   # wts:[1,2,1], mu:[1,2,1], var:[1,2,1,1]

        self.__update_wts(wts)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        #                         [np,2,1]                   [1,2,1]
        weighted_log_prob = self._estimate_log_prob(x[:,:,0:1], id_=0) + \
                            self._estimate_log_prob(x[:, :, 1:2], id_=1) + \
                            self._estimate_log_prob(x[:, :, 2:3], id_=2) + \
                            self._estimate_log_prob(x[:, :, 3:4], id_=3) + \
                            self._estimate_log_prob(x[:, :, 4:5], id_=4) + \
                            np.log(self.wts)   # [np,2,1]
        per_sample_score = logsumexp(weighted_log_prob, axis=1)      # [np,1]

        if as_average:
            return np.mean(per_sample_score)
        else:
            return np.squeeze(per_sample_score)  # [np,]

    def __update_mu(self, mu):
        assert np.shape(mu) in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if np.shape(mu) == (self.n_components, self.n_features):
            self.mu = mu[np.newaxis,:,:].copy()
        elif np.shape(mu) == (1, self.n_components, self.n_features):
            self.mu = mu.copy()

    def __update_var(self, var):
        if self.covariance_type == "full":
            if np.shape(var) == (1, self.n_components, self.n_features):
                self.var = var.copy()

    def __update_wts(self, wts):
        assert np.shape(wts) in [(1, self.n_components, 1)], "Input wts does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.wts = wts.copy()

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        if len(np.shape(x)) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = np.linalg.norm((x[:, np.newaxis, :].repeat(n_centers, 1) - tmp_center), ord=2, axis=2)
            l2_cls = np.argmin(l2_dis, axis=1)

            cost = 0
            for c in range(n_centers):
                cost += np.linalg.norm(x[l2_cls == c] - tmp_center[c], ord=2, axis=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = np.linalg.norm((x[:,np.newaxis,:].repeat(n_centers, 1) - center), ord=2, axis=2)
            l2_cls = np.argmin(l2_dis, axis=1)
            center_old = center.copy()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(axis=0)

            delta = np.linalg.norm((center_old - center), ord=2, axis=1).max()

        return (center[np.newaxis,:,:] * (x_max - x_min) + x_min)

    def bic(self, x):
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1
        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic


def fit_Guassian_processes(lat_fts, phy_fts):
    time, group = np.array(phy_fts['time']), np.array(phy_fts['group'])

    # here only analyze one group of embryoids (as an example)
    lat_fts_gp = lat_fts[np.where(group == 1)].copy()     # group 1c
    time = time[np.where(group == 1)].copy()
    time[time==52] = 54   # group 1
    tps_gp = np.unique(time)

    # fit two Gaussian processes
    n_comp = 2   # The number of Guassian components is selected based on the BIC creteria
    wts_ = np.zeros((len(tps_gp), n_comp), float)
    mus_ = np.zeros((len(tps_gp), n_comp, 5), float)
    vars_ = np.zeros((len(tps_gp), n_comp, 5), float)

    for ii, time_i in enumerate(tps_gp):
        data_ = lat_fts_gp[np.where(time==time_i)]
        GMM = GaussianMixture(n_components=n_comp, n_features=5)
        GMM.fit(x=data_)

        wts_[ii] = GMM.wts[0,:,0]
        mus_[ii] = GMM.mu[0,:,:]
        vars_[ii] = GMM.var[0,:,:]

    # predicted probability distributions of 5 latent features, at discreate time points (tps_gp)
    x_mesh = np.linspace(-1.5, 1.5, 500)
    dists = np.zeros((len(tps_gp), 5, 500), float)
    for ii, time_i in enumerate(tps_gp):
        for dim_i in range(5):
            for comp_i in range(n_comp):
                dists[ii, dim_i, :] += wts_[ii,comp_i]/np.sqrt(2*np.pi*vars_[ii,comp_i,dim_i]) * np.exp(-(x_mesh-mus_[ii,comp_i,dim_i])**2/2/vars_[ii,comp_i,dim_i])
            dists[ii,dim_i,:] = dists[ii,dim_i,:]/np.sum(dists[ii,dim_i,:])

    # # visualization (lat_fts distribution & distribution predicted by the GMM model)
    # x_mesh = np.linspace(-1.5, 1.5, 500)
    # for ii, time_i in enumerate(tps_gp):
    #     fig, ax1 = plt.subplots(figsize=(14, 1.7))
    #     ax2 = ax1.twinx()
    #     data_ = lat_fts_gp[np.where(time == time_i)]
    #     for dim_i in range(5):
    #         ax1.hist(data_[:, dim_i]+ dim_i*3, density=True, bins=20, color='grey')
    #         ax1.set_ylim(0, 14)
    #         prb_hat = np.zeros((500,), float)
    #         for comp_i in range(n_comp):
    #             wts_hat = wts_[ii, comp_i]
    #             mu_hat = mus_[ii, comp_i, dim_i]
    #             var_hat = vars_[ii, comp_i, dim_i]
    #             temp = wts_hat / np.sqrt(2 * np.pi * var_hat) * np.exp(-(x_mesh - mu_hat) ** 2 / 2 / var_hat)
    #             prb_hat += temp
    #             ax2.plot(x_mesh+dim_i*3, temp/np.sum(temp)*wts_hat, c='black', linewidth=1.5)
    #         ax2.plot(x_mesh+dim_i*3, prb_hat/np.sum(prb_hat), c='red', linewidth=3, alpha=0.5)
    #         ax2.set_ylim(bottom=0)
    #         ax2.set_ylim(0, 0.045)
    #     ax1.set_xlim(-1.5, 13.5)
    #     ax1.set_xticks([-1.5, 1.5, 4.5, 7.5, 10.5, 13.5])
    #     ax1.set_xticklabels([])
    #     ax1.set_yticks([0, 5, 10])
    #     ax1.set_yticklabels([0, 5, 10], fontsize=14)
    #     ax2.set_yticks([0, 0.02, 0.04])
    #     ax2.set_yticklabels([0, 0.02, 0.04], fontsize=14)
    #     ax1.set_xlabel('')
    #     ax1.set_ylabel('Density', fontsize=18)
    #     ax2.set_ylabel('Probability', fontsize=18)
    #     plt.tight_layout()
    #     plt.show()

    return tps_gp, wts_, mus_, vars_, dists

