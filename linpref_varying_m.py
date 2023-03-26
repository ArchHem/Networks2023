from network_OOP import *

N_r = 10
m = [2**i for i in range(1,5)]
N_t = 20000
lfig, lax = plt.subplots()

colors = ['red','gold','orchid']
def inf_distr(k,m):

    y = 2*m*(m+1)/(k*(k+1)*(k+2))

    return y

def color(i):

    return (0.2,1-i/len(m),i/len(m))

for i in range(len(m)):
    xmax, avg_y, std_y, mean_max, mean_std = system_averager(N_r,N_t,m[i],scale=1.2)
    lax.errorbar(xmax,avg_y,yerr = std_y,
                 label='Measured distribution with m = %.0f, T = %.0f with %.0f realizations' %(m[i],N_t,N_r),
                 marker='x',capsize = 2.5,ls='None',ms=10,color=color(i))
    lax.axvline(mean_max,lw=1,ls='--',color=color(i))
    pltx = np.linspace(m[i],xmax.max(),10000)
    lax.plot(pltx,inf_distr(pltx,m[i]),lw=1.5,ls='--',label=r'$P_{\infty}(k,m)$'+', m=%.0f'%m[i],color=color(i))

lax.set_yscale('log')
lax.set_xscale('log')
lax.set_xlabel(r'$k$')
lax.set_ylabel('Normalized frequencies')
lax.legend()
lax.grid()
plt.show()