from network_OOP import *

"""Test a few simple cases, beginning with connection a new edge"""
netw1 = Network([[1],[0]])
netw1.append_vertex()
"""Connect it to second vertex"""
netw1.estabilish_connection(1,connected_id=2)
"""Network now has 2 edges - count them"""
print('Number of edges is: %.0f' %netw1.get_edge_count()+', as expected.')

"""Statistical test - starting networks will yield roughly the same distribution in the infinite time limit under a 
linear attachment preference. 
To show two extremes, we do a fully connected initial network and a network where each edge connects to a single node 
and we perform a t-test.
"""

tfig, tax = plt.subplots()
m = 3
Nt = 40000

cutoff = 200
val, xt, kta, kto, xw, kwa, kwo = tallwide_t_test(10,Nt,m,cutoff=cutoff,tall_num=2*m)
print('The approximate bin-wise p-values are: ', val)

tax.errorbar(xt,kta,ls='None',color='green',marker='x',yerr=kto,capsize = 3,label='Initially tall network')
tax.errorbar(xw,kwa,ls='None',color='red',marker='x',yerr=kwo,capsize = 3,label='Initially wide network, m')
tax.set_yscale('log')
tax.set_xscale('log')
tax.axvline(cutoff,ls='--',label='Maximum considered degree')
tax.grid()
tax.legend()
tax.set_xlabel(r'$k$')
tax.set_ylabel('Normalized frequencies')

"""
As we can see, the two distributions are roughly the same, albeit the tails have a slight breakdown where we can see
the bias of the initially 'tall' network. A naive Pearson's P-test can also confirm the results.
"""
"""
Thus, both networks converged to the same distribution within reasonable limits. (its reasonable to reject the base 
hypothesis at a significance level of 0.05)
"""

"""Next, we test if a double edge is possible to occur ot not. We do this via propegating a 'wide' network in time."""

md = 10
NTD = 10000
isdouble = generate_fully_connected(md)
isdouble.lin_pref_time_evul(NTD,md)

state_list = isdouble.state

N_vertices = len(state_list)
connection_matrix = np.zeros((N_vertices,N_vertices),dtype='int32')

for i in range(N_vertices):
    for j in state_list[i]:
        connection_matrix[i,j] = connection_matrix[i,j] + 1

print('Are there only simple connections? ',np.amax(connection_matrix) ==1)

plt.show()

