from network_OOP import *

m1 = 2
m2 = 4
m3 = 8
m4 = 16
m5 = 32

N_t = 50000

netm_1 = generate_fully_connected(m1)
netm_1.lin_pref_time_evul(N_t,m1)

netm_2 = generate_fully_connected(m2)
netm_2.lin_pref_time_evul(N_t,m2)

netm_3 = generate_fully_connected(m3)
netm_3.lin_pref_time_evul(N_t,m3)

netm_4 = generate_fully_connected(m4)
netm_4.lin_pref_time_evul(N_t,m4)

netm_5 = generate_fully_connected(m5)
netm_5.lin_pref_time_evul(N_t,m5)

k1 = netm_1.edge_list_gen()
k2 = netm_2.edge_list_gen()
k3 = netm_3.edge_list_gen()
k4 = netm_4.edge_list_gen()
k5 = netm_5.edge_list_gen()

def theoretical_distr(k,m,N):

    AC, AO = norm_factors(N,m)

    val = AC/(k*(k+1)*(k+2))

    return val

def to_cont(a,subdiv=10000):
    var = np.linspace(np.amin(a),np.amax(a),subdiv)
    return var

s = 1.2

x1, y1, bin1 = logbin(k1,scale=s)
x2, y2, bin2 = logbin(k2,scale=s)
x3, y3, bin3 = logbin(k3,scale=s)
x4, y4, bin4 = logbin(k4,scale=s)
x5, y5, bin5 = logbin(k5,scale=s)

locfig, locax = plt.subplots()

locax.set_yscale('log')
locax.set_xscale('log')
locax.scatter(x1,y1,label='m = %.0f'%m1, marker='x',color='blue',s=10)
locax.scatter(x2,y2,label='m = %.0f'%m2, marker='x',color='cyan',s=10)
locax.scatter(x3,y3,label='m = %.0f'%m3, marker='x',color='green',s=10)
locax.scatter(x4,y4,label='m = %.0f'%m4, marker='x',color='orange',s=10)
locax.scatter(x5,y5,label='m = %.0f'%m5, marker='x',color='red',s=10)

locax.plot(to_cont(k1),theoretical_distr(to_cont(k1),m1,N_t),label='P(k), m = %.0f'%m1, ls='--',color='blue',lw=0.7)
locax.plot(to_cont(k2),theoretical_distr(to_cont(k2),m2,N_t),label='P(k), m = %.0f'%m2, ls='--',color='cyan',lw=0.7)
locax.plot(to_cont(k3),theoretical_distr(to_cont(k3),m3,N_t),label='P(k), m = %.0f'%m3, ls='--',color='green',lw=0.7)
locax.plot(to_cont(k4),theoretical_distr(to_cont(k4),m4,N_t),label='P(k), m = %.0f'%m4, ls='--',color='orange',lw=0.7)
locax.plot(to_cont(k5),theoretical_distr(to_cont(k5),m5,N_t),label='P(k), m = %.0f'%m5, ls='--',color='red',lw=0.7)

locax.grid()
locax.legend()
locax.set_xlabel('k')
locax.set_ylabel('Normalized frequency')
plt.show()
