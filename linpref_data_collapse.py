from network_OOP import *

m = 4
N_t = 1000*np.array([2**i for i in range(5)])

colors = ['green','lime','cyan','blue','purple','crimson']


locfig, locax = plt.subplots()

def theory_inf(k,m):
    return 2*m*(m+1)/(k*(k+1)*(k+2))

def plotter(m,N_t,N_runs,axis,color='green'):
    xmax1, avg_y1, std_y1, mean_max1, mean_std1 = system_averager(N_runs,N_t,m)

    axis.errorbar(xmax1/mean_max1,avg_y1/theory_inf(xmax1,m),yerr = std_y1/theory_inf(xmax1,m),capsize = 5,color=color,ls='None',marker = 'x',
                  xerr=mean_std1/mean_max1**2,
                  label='Numerically re-scaled mean distribution for m=%.0f and T=%.0f, %.0f realizations.'
                  %(m,N_t,N_runs))

NR = 10
for i in range(len(N_t)):
    plotter(m,N_t[i],NR,locax,color=colors[i])

k_plot = np.linspace(0.01,1,1000)
locax.grid()
locax.set_yscale('log')
locax.set_xscale('log')
locax.set_ylabel(r"$\frac{P(k')}{P(k')_{\infty}}$")
locax.set_xlabel(r"$k'=\frac{k}{k_{1}}}$")

locax.legend()
plt.show()