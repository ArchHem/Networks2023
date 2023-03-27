from network_OOP import *


NR = 10
NT = 30000
m = np.array([3**i for i in range(1,3)])
r = (m/3).astype('int64')
scale = 1.2

hfig, hax = plt.subplots()
lfig, lax = plt.subplots()
nfig, nax = plt.subplots()

def color(i):

    return (0.2,1-i/len(m),i/len(m))

for i in range(len(m)):
    maxx, avgy, avgy_std, maxkavg, maxk_std, fbins = hybrid_system_averager(NR,NT,m[i],r[i],scale)
    lax.errorbar(m[i],maxkavg,yerr=maxk_std,capsize=5,ls='None',color=color(i),
                 label='Maximum detected mean degree for existing vertices model, m=%.0f' %m[i],marker='x')
    nax.errorbar(maxx,avgy,yerr=avgy_std,capsize=5,ls='None',marker='x',color=color(i),
                 label='Measured existing vertices degree distribution, m = %.0f, r = %.0f'%(m[i],r[i]))
    plotx = np.linspace(r[i],maxx.max(),10000)
    nax.plot(plotx,hybrid_distribution(plotx,m[i],r[i]),lw=2,color=color(i),
             label='Theoretical distribution for m = %.0f and r = %.0f'%(m[i],r[i]),ls='--')

    hax.errorbar(maxx/maxkavg,avgy/hybrid_distribution(maxx,m[i],r[i]),yerr = avgy_std/hybrid_distribution(maxx,m[i],r[i]),
                 xerr=maxk_std/maxkavg**2,ls='None', marker='x',capsize = 5, color=color(i),
                 label='Data collapse, existing vertices degree distribution, m = %.0f'%m[i])

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
nax.set_ylabel("Normalized event frequencies")
nax.set_xlabel(r"$k$")

plt.show()


