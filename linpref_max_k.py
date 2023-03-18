from network_OOP import *


"""We choose m = 4 as it allows us to get away with relatively small number of runs."""

mloc = 4
N_avg = 30
N_t = 1000*np.arange(1,5+1,1,dtype='int64')**2

locfig, locax = plt.subplots()

mean_k, mean_k_std = [], []
for i in N_t:
    locmean, locstd = max_k_numeric(i,mloc,N_avg)
    mean_k.append(locmean)
    mean_k_std.append(locstd/np.sqrt(N_avg))

mean_k = np.array(mean_k)
mean_k_std = np.array(mean_k_std)

plot_t = np.linspace(np.amin(N_t),np.amax(N_t),10000)
theoretical_max = max_mean_k(plot_t,mloc)

locax.errorbar(N_t,mean_k,yerr=mean_k_std,ls='None',capsize=6,marker='x',color='red',
               label='Numerically measured mean max k')
locax.plot(plot_t,plot_t,lw=0.7,color='red',ls='--',label='Finite, sum-based model')
locax.set_ylabel('Max mean k')
locax.set_xlabel('Running time (iterations)')
locax.set_yscale('log')
locax.set_xscale('log')
locax.grid()
locax.legend()
plt.show()



