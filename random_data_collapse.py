from network_OOP import *

kfig, kax = plt.subplots()

cfig, cax = plt.subplots()


m = 5
N_runs = 10
N_t = 1000*np.array([2**i for i in range(6)])

maxfound = []
maxfound_std = []
for i in range(len(N_t)):
    lx, avgy, stdy, maxk, maxk_std = random_system_averager(N_runs, N_t[i], m)
    cax.errorbar(lx/maxk,avgy/theoretical_random(lx,m),yerr=stdy/theoretical_random(lx,m),xerr=maxk_std/maxk**2,ls='None',
                 label = 'Data collapse for m = %.0f, T = %0.f, %.0f realizations'
                  %(m,N_t[i],N_runs),marker='x',capsize=5)
    maxfound.append(maxk)
    maxfound_std.append(maxk_std)

maxfound = np.array(maxfound)
maxfound_std = np.array(maxfound_std)
"""color=(0.2,1-i/len(N_t),i/len(N_t))"""
cax.grid()
cax.legend()
cax.set_yscale('log')
cax.set_xscale('log')
cax.set_ylabel(r"$\frac{P(k')}{P_{\infty}(k')}$")
cax.set_xlabel(r"$k'=\frac{k}{k_{1}}}$")

kax.errorbar(N_t,maxfound,yerr=maxfound_std, color='red',capsize=5,
             label='Mean, maximum degree as a function of T',marker='x',ls='None')

plot_T = np.linspace(np.amin(N_t),np.amax(N_t),1000)
kax.plot(plot_T,random_sum_maximum_degree(plot_T,m),lw=2.0,ls='--',color='green',label='Sum-based maximum degree prediction',
         )

kax.plot(plot_T,random_int_maximum_degree(plot_T,m),lw=2.0,ls='--',color='blue',label='Integral-based maximum degree prediction',
         )


kax.grid()
kax.legend()
kax.set_ylabel(r'$k_1$')
kax.set_xlabel('T')
kax.set_yscale('log')
kax.set_xscale('log')
plt.show()
