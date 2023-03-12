import numpy as np
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

    def estabilish_connection(self,conn_id,connected_id=True):
        if connected_id:
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
                                           p=np.asarray(local_edge_count)/total_edges_double)

        for j in correct_indices:
            self.estabilish_connection(j)

    def random_connecter(self,m):
        self.append_vertex()
        index_list = [j for j in range(len(self.state)-1)]
        correct_indices = np.random.choice(index_list, size=m)
        for j in correct_indices:
            self.estabilish_connection(j)

    def lin_pref_time_evul(self,NT,m):
        for t in tqdm(range(NT)):
            self.lin_pref_connecter(m)

    def random_time_evul(self,Nt,m):
        for t in tqdm(range(Nt)):
            self.random_connecter(m)


def generate_fully_connected(m):

    state_list = []

    for i in range(m):
        loclist = [j for j in range(m)]
        del loclist[i]
        state_list.append(loclist)

    network = Network(state_list)
    return network

def p_value(chi,ddof):

    return scstat.chi2.sf(chi,ddof)

def max_mean_k(N,m):

    return (-3+np.sqrt(1+4*N*m*(m+1)))/2



def norm_factors(N_time,m):

    k1 = max_mean_k(N_time,m)

    m_term = 2/(m+2)

    mean_upper_sum = (k1**2 + 3*k1 - m*(m+3))/(2*(k1+1)*(k1+2)*(m+1)*(m+2))

    A = (1-m_term)/mean_upper_sum

    return A, 2*m*(m+1)

def preferential_test(N_runs,N_time,m,a=1.2,cutoff = False,axplot = plt):

    k_arr = []
    fbin = []
    rx = []


    for i in range(N_runs):
        model = generate_fully_connected(m)
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
        binindices = range(int(fbin[i]),int(fbin[i+1]))
        expected_freq.append(np.sum(locdistr(np.array(binindices)))/max(binindices[-1]-binindices[0]+1,1))

    expected_freq = np.array(expected_freq)
    with np.errstate(divide='ignore', invalid='ignore'):
        to_sum = ((expected_freq-mean_event_freq)**2 / mean_event_seom**2)[mean_event_seom!=0]



    if cutoff!=False:
        to_sum = to_sum[rx[mean_event_seom!=0]<cutoff]
        axplot.axvline(cutoff,ls='--',color='blue',label='Maximum considered degree')

    N_params = 2

    chi2 = np.sum(to_sum)

    print(chi2)

    print(expected_freq[rx < cutoff].shape[0])

    red_chi2 = chi2/(expected_freq[rx < cutoff].shape[0]-1-N_params)

    axplot.scatter(rx,expected_freq,color='green',label='Predicted event frequencies',s=25)
    axplot.errorbar(rx,mean_event_freq,color='red',
                    label='Measured event frequencies',ls='None',yerr=mean_event_seom,lw=1.2,
                    marker='x',capsize=3)
    axplot.plot(rx,2*m*(m+1)/(rx*(rx+1)*(rx+2)),
                label='Predicted probability extended to continuous domain',color='orange',
                lw=1.2)
    axplot.yscale('log')
    axplot.xscale('log')
    axplot.grid()
    axplot.legend()

    P_value = p_value(red_chi2,2)

    return P_value




N_t = 40000
m = 3


print(preferential_test(20,N_t,m,cutoff=55))
"""
norm_constant, A0 = norm_factors(N_t,5)

model = generate_fully_connected(m)
model.lin_pref_time_evul(N_t,m)

ks = np.array(model.edge_list_gen())
x,y,b = logbin(ks,scale=1.2)
plt.plot(x,y,label='Measured distribution',color='red')

kcont = np.arange(0,10)

plt.plot(x,A0/(x*(x+1)*(x+2)),label="Estimated 'infinite' distribution",color='green',ls='--')
ld = np.amax(ks)

plt.plot(x,norm_constant /(x*(x+1)*(x+2)),label='Finite-scale normalization',ls='--')

plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.grid()
"""
plt.show()






