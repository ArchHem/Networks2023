import numpy as np
import scipy.stats
from tqdm import tqdm
from random import choice
import matplotlib.pyplot as plt
from logbin_2020 import logbin
import scipy.stats as scstat

class Network:
    def __init__(self,init_state):
        self.state = init_state
        self.edge_count = int(sum( [ len(listElem) for listElem in self.state])/2)
        self.vertices_count = len(self.state)

    def get_edge_count(self):

        return int(sum( [ len(listElem) for listElem in self.state])/2)

    def get_vertice_count(self):
        return len(self.state)

    def append_vertex(self):
        self.state.append([])
        self.edge_count = self.get_edge_count()
        self.vertices_count = self.get_vertice_count()

    def edge_list_gen(self):

        edges_count = [len(length) for length in self.state]

        return edges_count

    def estabilish_connection(self,conn_id,connected_id=None):
        if connected_id == None:
            connected_id = len(self.state) - 1
        self.state[connected_id].append(conn_id)
        self.state[conn_id].append(connected_id)
        self.edge_count = self.get_edge_count()
        self.vertices_count = self.get_vertice_count()

    def lin_pref_connecter(self,m):
        self.append_vertex()
        total_edges_double = 2*self.get_edge_count()
        local_edge_count = self.edge_list_gen()[:-1]

        index_list = [j for j in range(len(self.state)-1)]

        """use np.random.choice with size = m and p = local_edge_count/totaL_edges_double"""
        correct_indices = np.random.choice(index_list,size=m,
                                           p=np.asarray(local_edge_count)/total_edges_double,replace=False)

        for j in correct_indices:
            self.estabilish_connection(j)

    def is_connected(self,vfrom,vto):
        state = vfrom in self.state[vto]
        return state

    def lin_pref_time_evul(self,NT,m):
        for t in tqdm(range(NT)):
            self.lin_pref_connecter(m)

    def generate_random(self,E):
        N = len(self.state)
        indeces = np.arange(0,(N*(N-1)/2),1)
        bool_index = np.random.choice(indeces,size=(E,),replace=False).astype('int64')
        connected_ = np.zeros((int(N*(N-1)/2)))
        connected_[bool_index] = 1

        out = np.zeros((N, N))
        inds = np.triu_indices(len(out),k=1)
        out[inds] = connected_
        out = out.astype('int16')
        correct_indices = np.where(out == 1)
        """generate network via estabilishing connections"""

        for i in range(correct_indices[0].shape[0]):
            self.estabilish_connection(correct_indices[0][i],correct_indices[1][i])









def generate_fully_connected(m):
    state_list = []

    for i in range(m):
        loclist = [j for j in range(m)]
        del loclist[i]
        state_list.append(loclist)

    network = Network(state_list)
    return network

def generate_empty(N):
    state = [[] for i in range(N)]
    network = Network(state)
    return network

def generate_tall(m):

    state_list = [[i for i in range(1,m)]]

    for j in range(1,m):
        state_list.append([0])

    network = Network(state_list)
    return network

def p_value(chi,ddof):

    return scstat.chi2.sf(chi,ddof)

def max_mean_k(N,m):

    return m*N**(0.5)

def norm_factors(N_time,m):

    k1 = max_mean_k(N_time,m)

    m_term = 2/(m+2)

    mean_upper_sum = (k1**2 + 3*k1 - m*(m+3))/(2*(k1+1)*(k1+2)*(m+1)*(m+2))

    A = (1-m_term)/mean_upper_sum

    return A, 2*m*(m+1)

def preferential_test(N_runs,N_time,m,a=1.2,cutoff = False,axplot = plt,init='wide'):

    k_arr = []
    fbin = []
    rx = []

    for i in range(N_runs):
        if init=='wide':
            model = generate_fully_connected(m)
        else:
            model = generate_tall(m)
        model.lin_pref_time_evul(N_time,m)
        loc_ks = model.edge_list_gen()
        x, event_freq, bins = logbin(loc_ks,scale=a,actual_zeros=False)
        if len(bins) > len(fbin):
            fbin = bins
            rx = x
        k_arr.append(event_freq)


    longest = max([j.shape[0] for j in k_arr])

    for j in range(len(k_arr)):
        k_arr[j] = np.hstack((k_arr[j],np.zeros(longest-k_arr[j].shape[0])))

    k_arr = np.array(k_arr)
    mean_event_freq = k_arr.mean(axis=0)
    mean_event_seom = k_arr.std(axis=0)/np.sqrt(N_runs)

    def locdistr(k):

        AC, A0 = norm_factors(N_time,m)

        k1 = np.zeros_like(k)

        k2 = np.where(k > m, AC/(k*(k+1)*(k+2)),k1)

        k3 = np.where(k == m, 2/(m+2),k2)

        return k3

    expected_freq = []
    for i in range(len(fbin)-1):
        binindices = np.array(range(int(fbin[i]),int(fbin[i+1]+1)))
        expected_freq.append(np.sum(locdistr(np.array(binindices[:-1])))/(binindices[-1]-binindices[0]))

    expected_freq = np.array(expected_freq)
    with np.errstate(divide='ignore', invalid='ignore'):
        to_sum = ((expected_freq-mean_event_freq)**2 / mean_event_seom**2)[mean_event_seom!=0]



    if cutoff!=False:
        to_sum = to_sum[rx[mean_event_seom!=0]<cutoff]
        axplot.axvline(cutoff,ls='--',color='blue',label='Maximum considered degree')

    N_params = 2

    chi2 = np.sum(to_sum)

    print('Chi-square value: ',chi2)

    df = expected_freq[rx < cutoff].shape[0]-N_params-1

    print('Degrees of freedom: ', df)

    axplot.scatter(rx,expected_freq,color='green',label='Predicted event frequencies',s=25)
    axplot.errorbar(rx,mean_event_freq,color='red',
                    label='Measured event frequencies',ls='None',yerr=mean_event_seom,lw=1.2,
                    marker='x',capsize=3)
    axplot.plot(rx,2*m*(m+1)/(rx*(rx+1)*(rx+2)),
                label='Predicted probability extended to continuous domain, m=%.0f' %(m),color='orange',
                lw=1.2)
    axplot.set_yscale('log')
    axplot.set_xscale('log')
    axplot.grid()
    axplot.legend()
    P_value = p_value(chi2,df)

    return P_value

def tallwide_t_test(N_runs,N_time,m,a=1.2,cutoff = 55,axplot = plt):
    k_arrw = []
    k_arrt = []
    fbin = []
    rx = []

    for i in range(N_runs):
        model_wide = generate_fully_connected(m)
        model_tall = generate_tall(m)
        model_tall.lin_pref_time_evul(N_time, m)
        model_wide.lin_pref_time_evul(N_time, m)
        loc_kst = model_tall.edge_list_gen()
        loc_ksw = model_wide.edge_list_gen()
        xt, event_freqt, binst = logbin(loc_kst, scale=a, actual_zeros=False)
        xw, event_freqw, binsw = logbin(loc_ksw, scale=a, actual_zeros=False)

        if max(len(binst),len(binsw)) > len(fbin):
            pos = [binst,binsw]
            poslen = [len(binst),len(binsw)]
            fbin = pos[np.argmax(np.array(poslen))]
            xlis = [xt,xw]
            rx = xlis[np.argmax(np.array(poslen))]
        k_arrw.append(event_freqw)
        k_arrt.append(event_freqt)

    longest = max([j.shape[0] for j in k_arrw]+[i.shape[0] for i in k_arrt])


    for j in range(len(k_arrw)):
        k_arrw[j] = np.hstack((k_arrw[j], np.zeros(longest - k_arrw[j].shape[0])))

    for j in range(len(k_arrt)):
        k_arrt[j] = np.hstack((k_arrt[j],np.zeros(longest - k_arrt[j].shape[0])))

    k_arrt = np.array(k_arrt)
    k_arrw = np.array(k_arrw)

    closest_index = np.argmax(np.where(rx < cutoff,rx,-1))

    kt_avg = np.average(k_arrt,axis=0)
    kt_std = np.std(k_arrt,axis=0)/np.sqrt(N_runs)

    kw_avg = np.average(k_arrw,axis=0)
    kw_std = np.std(k_arrw,axis=0)/np.sqrt(N_runs)

    x_w = rx[0:len(kw_avg)]
    x_t = rx[0:len(kt_avg)]
    with np.errstate(divide='ignore', invalid='ignore'):
        t_test_val = np.abs(kt_avg[:closest_index+1]/np.sum(kt_avg[:closest_index+1])-
                        kw_avg[:closest_index+1]/np.sum(kw_avg[:closest_index+1]))/\
                        np.sqrt(kt_std[:closest_index+1]**2 + kw_std[:closest_index+1]**2)
    df = N_runs-1-2
    p_val = scstat.t.sf(t_test_val,df=df)

    return p_val, x_t, kt_avg, kt_std, x_w, kw_avg, kw_std

def max_k_numeric(Nt,m,N_runs):

    k1 = []
    for i in range(N_runs):
        locnet = generate_fully_connected(m)
        locnet.lin_pref_time_evul(Nt,m)
        max_k = np.amax(locnet.edge_list_gen())
        k1.append(max_k)

    k1 = np.array(k1)

    mean_max = np.average(k1)
    mean_std = np.std(k1)/np.sqrt(N_runs)

    return mean_max, mean_std

def



if __name__ == "__main__":
    N_t = 20000
    m = 3
    NR = 30
    cutoff = 55
    ofig, oax = plt.subplots()
    sfig, sax = plt.subplots()
    print(preferential_test(NR,N_t,m,cutoff=cutoff,a=1.2,axplot=sax))
    N = 10000
    E = 40000
    netw1 = generate_empty(N)
    netw1.generate_random(E)
    ks = netw1.edge_list_gen()
    p = E/(N*(N-1)/2)
    x,y,edges = logbin(ks,scale=1.2)
    mu = p*N
    plotx = np.arange(int(np.amin(x)),int(np.amax(x))+1,1,dtype='int64')
    oax.plot(plotx,scipy.stats.poisson.pmf(plotx,10000*p))
    oax.scatter(x,y)
    oax.set_yscale('log')
    oax.set_xscale('log')


plt.show()






