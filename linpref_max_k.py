from network_OOP import *
from scipy.optimize import curve_fit

"""We choose m = 3 as it allows us to get away with relatively small number of runs."""

mloc = 3
N_avg = 20
N_t = 1000*np.array([2**i for i in range(6)])

locfig, locax = plt.subplots()

mean_k, mean_k_std = [], []
for i in N_t:
    locmean, locstd = max_k_numeric(i,mloc,N_avg)
    mean_k.append(locmean)
    mean_k_std.append(locstd/np.sqrt(N_avg))

mean_k = np.array(mean_k)
mean_k_std = np.array(mean_k_std)

plot_t = np.linspace(np.amin(N_t),np.amax(N_t),500)
theoretical_max = precise_max_predictor(plot_t,mloc)
precise_theory = precise_int_predictor(plot_t,mloc)

def power_law(x,c,beta):
    return c*x**beta

locax.errorbar(N_t,mean_k,yerr=mean_k_std,ls='None',capsize=6,marker='x',color='red',
               label='Numerically measured mean max k, number of realizations is: %.0f' %N_avg)



locax.plot(plot_t, theoretical_max,lw=1.8,color='cyan',ls='--',label='Finite, sum-based prediction')
locax.plot(plot_t, mloc*plot_t**(0.5),color='green',ls='--',lw=1.8,label='Integral estimate based prediction')
locax.plot(plot_t, precise_theory,color='black',ls='--',lw=1.8,label='Higher-order integral estimate')

locax.set_ylabel('Max mean k')
locax.set_xlabel('Running time (iterations)')
locax.set_yscale('log')
locax.set_xscale('log')

chi2_sum = np.sum(((mean_k-precise_max_predictor(N_t,mloc))/mean_k_std)**2)
chi2_int = np.sum(((mean_k-mloc*N_t**0.5)/mean_k_std)**2)

print('Chi2 value of integral-based prediction: %.3f' %chi2_int)
print('Chi2 value of sum-based prediction: %.3f' %chi2_sum)
print('Degrees of freedom: %.0f'%(len(N_t)-3))

locax.grid()
locax.legend()
plt.show()



