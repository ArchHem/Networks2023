import numpy as np
import scipy.stats
from tqdm import tqdm
from random import choice
import matplotlib.pyplot as plt
from logbin_2020 import logbin
import scipy.stats as scstat
from scipy.optimize import root
import scipy.special as scspecial
plt.rcParams.update({'font.size': 16,'axes.labelsize': 28,
         'axes.titlesize': 32})

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

    def random_connecter(self,m):
        self.append_vertex()
        index_list = [j for j in range(len(self.state) - 1)]
        correct_indices = np.random.choice(index_list,size=m,replace=False)
        for j in correct_indices:
            self.estabilish_connection(j)

    def existing_connecter(self,m,r):
        self.append_vertex()
        index_list = [j for j in range(len(self.state) - 1)]
        correct_indices = np.random.choice(index_list, size=r, replace=False)
        total_edges_double = 2 * self.get_edge_count()
        local_edge_count = self.edge_list_gen()[:-1]

        for j in correct_indices:
            self.estabilish_connection(j)


        """do the linear preferential for the rest"""
        linpref_pairs = []
        while len(linpref_pairs) <(m-r):
            selected_index1 = np.random.choice(index_list,size=1,
                                           p=np.asarray(local_edge_count)/total_edges_double,replace=False)[0]

            selected_index2 = np.random.choice(index_list,size=1,
                                           p=np.asarray(local_edge_count)/total_edges_double,replace=False)[0]
            pair = {selected_index1,selected_index2}
            if pair not in linpref_pairs and len(pair)==2:
                linpref_pairs.append(pair)

        linpref_pairs = [list(i) for i in linpref_pairs]

        for i in range(len(linpref_pairs)):
            self.estabilish_connection(linpref_pairs[i][0],linpref_pairs[i][1])



    def is_connected(self,vfrom,vto):
        state = vfrom in self.state[vto]
        return state

    def lin_pref_time_evul(self,NT,m):
        for t in tqdm(range(NT)):
            self.lin_pref_connecter(m)

    def generate_random(self,NT,m):
        for t in tqdm(range(NT)):
            self.random_connecter(m)

    def hybrid_time_evul(self,NT,m,r):
        for t in tqdm(range(NT)):
            self.existing_connecter(m,r)

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

def precise_max_predictor(N,m):
    k1 = 0.5 * (np.sqrt(4*N*(m+1)*m+1)-1)

    return k1

def precise_int_predictor(N,m):
    A = 2*m*(m+1)

    def to_solve(k1):
        return A*N*(-np.log(k1+2)/2+np.log(k1+1)-np.log(k1)/2)-1

    solution = root(to_solve,x0=m*N**0.5)

    return solution.x

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
        if cutoff!=False:
            expected_freq = expected_freq/np.sum(expected_freq[rx < cutoff])
            mean_event_freq = mean_event_freq/np.sum(mean_event_freq[rx < cutoff])
        to_sum = ((expected_freq-mean_event_freq)**2 / mean_event_seom**2)[mean_event_seom!=0]



    if cutoff!=False:
        to_sum = to_sum[rx[mean_event_seom!=0]<cutoff]
        axplot.axvline(cutoff,ls='--',color='blue',label='Maximum considered degree')

    N_params = 2

    chi2 = np.sum(to_sum)

    print('Chi-square value: ',chi2)

    df = expected_freq[rx < cutoff].shape[0]-N_params-1

    print('Degrees of freedom: ', df)

    axplot.scatter(rx,expected_freq,color='green',
                   label='Predicted event frequencies for m= %.0f, T= %.0f, %.0f realizations' %(m,N_time,N_runs),s=25)
    axplot.errorbar(rx,mean_event_freq,color='red',
                    label='Measured event frequencies',ls='None',yerr=mean_event_seom,lw=1.2,
                    marker='x',capsize=3)

    yt = 2*m*(m+1)/(rx*(rx+1)*(rx+2))
    yt[rx <m] = 0
    axplot.plot(rx,yt,
                label='Predicted probability extended to continuous domain, m=%.0f' %(m),color='orange',
                lw=1.2)

    axplot.set_yscale('log')
    axplot.set_xscale('log')

    axplot.set_ylabel('Normalized mean event frequencies')
    axplot.set_xlabel(r'$k$')

    axplot.grid()
    axplot.legend()

    P_value = p_value(chi2,df)

    return P_value


def generate_ER(N,E):

    model = generate_empty(N)
    status = True
    count = 0
    connections = []
    while status:

        indices = np.arange(0,N,1,dtype='int')
        to_connect = np.random.choice(indices,size=2,replace=False)
        locset = set(to_connect.tolist())
        if locset not in connections:
            connections.append(locset)
            count = count + 1
        if count == E:
            status = False

    to_connect = [list(i) for i in connections]

    for j in to_connect:
        model.estabilish_connection(j[0],j[1])

    return model



def system_averager(N_runs, N_time, m, scale = 1.2):
    k1 = []
    ys = []
    xmax = []
    for i in range(N_runs):
        locnet = generate_fully_connected(m)
        locnet.lin_pref_time_evul(N_time, m)
        ks = locnet.edge_list_gen()
        max_k = np.amax(ks)
        k1.append(max_k)
        x,y,bins = logbin(ks,scale=scale,actual_zeros=False)
        if len(xmax) < len(x):
            xmax = x
        ys.append(y)

    for i in range(N_runs):
        diff = -len(ys[i])+len(xmax)
        if diff !=0:
            ys[i] = np.hstack((ys[i],np.zeros(diff,)))

    ys = np.array(ys)

    k1 = np.array(k1)

    mean_max = np.average(k1)
    mean_std = np.std(k1) / np.sqrt(N_runs)
    avg_y = np.mean(ys,axis=0)
    std_y = np.std(ys,axis=0)/np.sqrt(N_runs)

    return xmax, avg_y, std_y, mean_max, mean_std

def tallwide_t_test(N_runs,N_time,m,a=1.2,cutoff = 55,tall_num=None,axplot = plt):
    k_arrw = []
    k_arrt = []
    fbin = []
    rx = []

    if tall_num == None:
        tall_num = m

    for i in range(N_runs):
        model_wide = generate_fully_connected(m)
        model_tall = generate_tall(tall_num)
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
    closest_index = np.argmax(np.where(rx < cutoff,rx,-1))

    for j in range(len(k_arrw)):
        k_arrw[j] = np.hstack((k_arrw[j], np.zeros(longest - k_arrw[j].shape[0])))

    for j in range(len(k_arrt)):
        k_arrt[j] = np.hstack((k_arrt[j],np.zeros(longest - k_arrt[j].shape[0])))


    k_arrt = np.array(k_arrt)
    k_arrw = np.array(k_arrw)

    k_arrt = k_arrt/np.expand_dims(np.sum(k_arrt[:,:closest_index+1],axis=1),axis=0).T
    k_arrw = k_arrw / np.expand_dims(np.sum(k_arrw[:, :closest_index + 1],axis=1),axis=0).T


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
    """Test is two tailed - we dont know which group is 'better'. Use symetry."""
    p_val = 2*scstat.t.sf(t_test_val,df=df)

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

def theoretical_random(k,m):

    y = m**(k-m)/(1+m)**(1+k-m)
    y[k < m] = 0
    return y


def random_p_tester(N_runs,NT,m,axplot,a=1, cutoff = np.inf):

    lx = []
    fbin = []
    ys = []

    for j in range(N_runs):
        locnet = generate_fully_connected(m)
        locnet.generate_random(NT,m)
        lock = locnet.edge_list_gen()
        x, y, edges = logbin(lock,a,actual_zeros=False)
        if len(lx)<len(x):
            lx = x
            fbin = edges
        ys.append(y)


    maxlength = len(lx)
    for i in range(N_runs):
        diff = maxlength-len(ys[i])
        if diff != 0:
            ys[i] = np.hstack((ys[i],np.zeros((diff,))))

    avgy = np.mean(ys,axis=0)
    stdy = np.std(ys,axis=0)/np.sqrt(N_runs)

    """generate expected frequencies"""

    theoretical =[]

    for j in range(len(fbin)-1):

        indices = np.arange(fbin[j],fbin[j+1],1)
        length = fbin[j+1]-fbin[j]
        theoretical.append(np.sum(theoretical_random(indices,float(m)))/length)

    theoretical = np.array(theoretical)
    with np.errstate(divide='ignore', invalid='ignore'):
        to_sum = ((theoretical-avgy)/stdy)**2
    to_sum = to_sum[(to_sum < cutoff) & (stdy !=0)]
    chi2 = np.sum(to_sum)

    p = p_value(chi2,len(to_sum)-3)

    print('Chi2 = %.3f, degrees of freedom = %.0f' %(chi2,len(stdy)-3))

    axplot.errorbar(lx,avgy,ls='None', marker='x',capsize = 5, color='red',yerr = stdy,
                    label='Measured mean via binned data, T = %.0f and m = %.0f' %(NT,m))

    axplot.set_xlabel('k')
    axplot.set_ylabel('Normalized event frequency')


    axplot.scatter(lx,theoretical,s=20,color='green',label='Theoretical event'
                ' frequencies for T = %.0f and m = %.0f'%(NT,m),zorder = 20)
    if cutoff!=np.inf:
        axplot.axvline(cutoff,color='blue',ls='--')
    axplot.grid()
    axplot.legend()

    return p

def random_system_averager(NR,Nt,m,a=1.2):
    lx = []
    fbin = []
    ys = []
    maxes = []
    for j in range(NR):
        locnet = generate_fully_connected(m)
        locnet.generate_random(Nt, m)
        lock = locnet.edge_list_gen()
        x, y, edges = logbin(lock, a, actual_zeros=False)
        if len(lx) < len(x):
            lx = x
            fbin = edges
        ys.append(y)
        maxes.append(np.amax(lock))

    maxlength = len(lx)
    for i in range(NR):
        diff = maxlength - len(ys[i])
        if diff != 0:
            ys[i] = np.hstack((ys[i], np.zeros((diff,))))

    avgy = np.mean(ys, axis=0)
    stdy = np.std(ys, axis=0)/np.sqrt(NR)

    maxes = np.array(maxes)

    maxk = np.average(maxes)
    maxk_std = np.std(maxes)/np.sqrt(NR)

    return lx, avgy, stdy, maxk, maxk_std

def random_sum_maximum_degree(N,m):

    return m - np.log(N)/(np.log(m)-np.log(m+1))

def random_int_maximum_degree(N,m):

    def to_solve(k):

        return ((m+1)**(m-1)*np.exp(np.log(m)*k-np.log(m+1)*k))/(m**m*(np.log(m+1)-np.log(m)))-1/N

    sol = scipy.optimize.root(to_solve,random_sum_maximum_degree(N,m))

    return sol.x

def normalize(x,bins= None):
    if bins == None:
        return x/np.sum(x)
    else:
        return x/np.sum(x*np.diff(bins))


def hybrid_distribution(k,m,r):

    def to_exp(k,m,r):
        b = m / (m - r)
        return scspecial.loggamma(k+r*b) - scspecial.loggamma(k+1+r*b+b)

    A = (1/((m-r)*r/m + r +1))/np.exp(to_exp(r,m,r))


    return A*np.exp(to_exp(k,m,r))

def hybrid_system_averager(NR,NT,m,r,scale,m_init = None):
    if m_init == None:
        m_init = m

    maxx = []
    maxk = []
    ys = []
    fbins=[]

    for i in range(NR):
        model = generate_fully_connected(m_init)
        model.hybrid_time_evul(NT,m,r)
        kl = model.edge_list_gen()
        maxk.append(kl)
        x, y, bins = logbin(kl,scale =scale,actual_zeros=False)
        if len(x)>len(maxx):
            maxx = x
            fbins = bins
        ys.append(y)

    for i in range(NR):
        diff = len(maxx)-len(ys[i])
        if diff!=0:
            ys[i] = np.hstack((ys[i],np.zeros((diff))))

    ys = np.array(ys)
    maxk = np.array(maxk)

    avgy = np.average(ys,axis=0)
    avgy_std = np.std(ys,axis=0)/np.sqrt(NR)

    maxkavg = np.average(maxk)
    maxk_std = np.std(maxk)/np.sqrt(NR)

    return maxx, avgy, avgy_std, maxkavg, maxk_std,fbins

"""
m = 3
r = 1
model = generate_fully_connected(10)
model.hybrid_time_evul(60000,m,r)
kl = model.edge_list_gen()
x,y,bins = logbin(kl,actual_zeros=True)
plt.plot(x,y)
plt.plot(x,hybrid_distribution(x,m,r))
plt.xscale('log')
plt.yscale('log')
plt.show()"""



