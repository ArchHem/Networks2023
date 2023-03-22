from network_OOP import *

"""This field is meant to show the effects of varying E and N, and its effects on the P-value"""


locfig, locax = plt.subplots()

def loc_distr(k, NV, E):
    k = k.astype('int64')
    p = E / ((NV - 1) * NV / 2)
    y = scstat.binom.pmf(k, NV, p)
    return y

E = 1000*np.array([2**i for i in range(2,5)])
N = np.array([5000,20000])


for i in range(len(E)):
    for j in range(len(N)):
        locmodel = generate_empty(N[j])
        locmodel.generate_random(E[i])
        kl = locmodel.edge_list_gen()
        xl, yl, binl = logbin(kl,scale=1.0,actual_zeros=False)
        locax.scatter(xl,yl,marker = 'x',color = (1,1-j/len(E),1-i/len(N)),label='N = %.0f, E = %.0f' %(N[j],E[i]))
        locax.plot(xl, loc_distr(xl.astype('int'), N[j], E[i]),
                      ls='--', color=(1, 1 - j / len(E), 1 - i / len(N)),
                   label= 'Binomial for N = %.0f, E = %.0f' %(N[j],E[i]))


locax.set_xlabel('k')
locax.set_ylabel('Normalized frequencies.')
locax.grid()
locax.legend()
plt.show()

