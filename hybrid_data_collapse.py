from network_OOP import *


NR = 2
NT = 10000
m = np.array([3**i for i in range(1,4)])
r = m/3
scale = 1.2

hfig, hax = plt.subplots()
lfig, lax = plt.subplots()
nfig, nax = plt.subplots()


for i in range(len(m)):
    maxx, avgy, avgy_std, maxkavg, maxk_std, fbins = hybrid_system_averager(NR,NT,m[i],r[i],scale)
    lax.errorbar(m[i],maxkavg,yerr=maxk_std,capsize=5,ls='None',color=(0.5,1-i/len(m),i/len(m)),
                 label='Maximum detected mean degree for existing vertices model, m=%.0f' %m[i],marker='x')
    nax.errorbar(maxx,avgy,yerr=avgy_std,capsize=5,ls='None',marker='x',
                 label='Existing vertices degree distribution, m = %.0f'%m[i])
    hax.errorbar(maxx/maxkavg,yerr = maxk_std/normalize(hybrid_distribution(maxx,m[i],r[i])),
                 xerr=maxk_std/maxkavg**2,ls='None', marker='x',
                 label='Data collapsed, existing vertices degree distribution, m = %.0f'%m[i])

lax.set_yscale('log')
lax.set_xscale('log')
lax.grid()
lax.legend()
lax.set_xlabel(r"$m$")
lax.set_ylabel(r"$k_1$")

hax.set_yscale('log')
hax.set_xscale('log')
hax.grid()
hax.legend()
hax.set_ylabel(r"$\frac{P(k')}{P_{\infty}(k')}$")
hax.set_xlabel(r"$k'=\frac{k}{k_{1}}}$")

nax.set_yscale('log')
nax.set_xscale('log')
nax.grid()
nax.legend()
nax.set_ylabel(r"$P(k)$")
nax.set_xlabel(r"$k$")

plt.show()

