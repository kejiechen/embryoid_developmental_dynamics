import numpy as np
from matplotlib import pyplot as plt
import pdb


def fit_mean_reverting_process(tps, wts, mus, vars, dists, n_comp, dt):
    # re-order wts, mus, vars
    arg_min = np.mean(np.argmin(mus, axis=1), axis=1)
    group_ids = np.array([0 if i < 0.5 else 1 for i in arg_min])
    wts_ = np.zeros((len(tps), n_comp), float)
    mus_ = np.zeros((len(tps), n_comp, 5), float)
    vars_ = np.zeros((len(tps), n_comp, 5), float)
    for ti in range(len(tps)):
        wts_[ti, 0], wts_[ti, 1] = wts[ti, group_ids[ti]], wts[ti, 1 - group_ids[ti]]
        mus_[ti, 0, :], mus_[ti, 1, :] = mus[ti, group_ids[ti], :], mus[ti, 1 - group_ids[ti], :]
        vars_[ti, 0, :], vars_[ti, 1, :] = vars[ti, group_ids[ti], :], vars[ti, 1 - group_ids[ti], :]

    # continuous latent feature evolution
    import scipy.interpolate
    T, dt, n_par, theta_1, theta_2 = int(tps[-1]-tps[0]), dt, 10**4, 100, 10
    lat_ft_evol = np.zeros((int(T/dt)+1, n_par, 5), float)
    for dim_i in range(5):
        func = scipy.interpolate.interp1d(np.linspace(-1.5,1.5,500), dists[0,dim_i,:], kind='quadratic')
        p_data = func(np.linspace(-1.5, 1.5, n_par))
        lat_ft_evol[0,:,dim_i] = np.random.choice(a=n_par, p=p_data/np.sum(p_data), size=n_par)*3.0/n_par - 1.5

        # run mean-reverting stochastic process
        for ti in range(1,int(T/dt)+1):
            id_r = np.where(tps[0]+ti*dt<=tps)[0][0]
            for pi in range(n_par):
                if np.random.rand() < wts_[id_r,0]:
                    lat_ft_evol[ti,pi,dim_i] = lat_ft_evol[ti-1,pi,dim_i] + theta_1*(mus_[id_r,0,dim_i]-lat_ft_evol[ti-1,pi,dim_i])*dt + theta_2*np.sqrt(vars_[id_r,0,dim_i]*dt)*np.random.normal(0)
                else:
                    lat_ft_evol[ti,pi,dim_i] = lat_ft_evol[ti-1,pi,dim_i] + theta_1*(mus_[id_r,1,dim_i]-lat_ft_evol[ti-1,pi,dim_i])*dt + theta_2*np.sqrt(vars_[id_r,1,dim_i]*dt)*np.random.normal(0)

            # visualization and verification: distributions predicted by the stochastic process, dists (input)
            if ti*dt+tps[0] in [i*1.0 for i in tps]:
                fig, ax = plt.subplots(figsize=(5,2.5))
                ax.hist(lat_ft_evol[ti,:,dim_i], density=True, bins=100, color='orange', alpha=0.7)
                func = scipy.interpolate.interp1d(np.linspace(-1.5, 1.5, 500), dists[np.where(tps==ti*dt+tps[0])[0][0],dim_i,:], kind='quadratic')
                p_data = func(np.linspace(-1.5, 1.5, n_par))
                lat_ft_from_dists = np.random.choice(a=n_par, p=p_data / np.sum(p_data), size=n_par)*3.0/p_data.size-1.5
                ax.hist(lat_ft_from_dists, density=True, bins=100, color='grey', alpha=0.5)
                plt.tight_layout()
                plt.show()
                pdb.set_trace()

        return lat_ft_evol


def lat_fts_propation(emb_fts, tps, wts, mus, vars, n_comp, dt):
    # re-order wts, mus, vars
    arg_min = np.mean(np.argmin(mus, axis=1), axis=1)
    group_ids = np.array([0 if i < 0.5 else 1 for i in arg_min])
    wts_ = np.zeros((len(tps), n_comp), float)
    mus_ = np.zeros((len(tps), n_comp, 5), float)
    vars_ = np.zeros((len(tps), n_comp, 5), float)
    for ti in range(len(tps)):
        wts_[ti, 0], wts_[ti, 1] = wts[ti, group_ids[ti]], wts[ti, 1 - group_ids[ti]]
        mus_[ti, 0, :], mus_[ti, 1, :] = mus[ti, group_ids[ti], :], mus[ti, 1 - group_ids[ti], :]
        vars_[ti, 0, :], vars_[ti, 1, :] = vars[ti, group_ids[ti], :], vars[ti, 1 - group_ids[ti], :]

    T, dt, n_par, theta_1, theta_2 = int(tps[-1]-tps[0]), dt, 20, 100, 10
    lat_ft_evol = np.zeros((int(T/dt)+1, n_par, 5), float)
    lat_ft_evol[0,:,:] = emb_fts[np.newaxis,:].repeat(n_par, axis=0)

    for dim_i in range(5):
        # run mean-reverting stochastic process
        for ti in range(1,int(T/dt)+1):
            id_r = np.where(tps[0]+ti*dt<=tps)[0][0]
            for pi in range(n_par):
                if np.random.rand() < wts_[id_r,0]:
                    lat_ft_evol[ti,pi,dim_i] = lat_ft_evol[ti-1,pi,dim_i] + theta_1*(mus_[id_r,0,dim_i]-lat_ft_evol[ti-1,pi,dim_i])*dt + theta_2*np.sqrt(vars_[id_r,0,dim_i]*dt)*np.random.normal(0)
                else:
                    lat_ft_evol[ti,pi,dim_i] = lat_ft_evol[ti-1,pi,dim_i] + theta_1*(mus_[id_r,1,dim_i]-lat_ft_evol[ti-1,pi,dim_i])*dt + theta_2*np.sqrt(vars_[id_r,1,dim_i]*dt)*np.random.normal(0)

    return lat_ft_evol


