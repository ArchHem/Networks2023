from network_OOP import *

"""This field is meant to show the effects of varying E and N, and its effects on the P-value"""


locfig, locax = plt.subplots()

NT = 1000*np.array([2**i for i in range(3,5)])
m = np.array([4,8,16])
NR = 10

for i in range(len(NT)):
    for j in range(len(m)):
        lx, avgy, stdy, maxk, maxk_std = random_system_averager(NR,NT[i],m[j])
        locax.errorbar(lx,avgy,marker = 'x',color = (0,1-j/len(m),1-i/len(NT)),
                       label='m = %.0f, T = %.0f, number of realizations = %.0f' %(m[j],NT[i],NR),
                       yerr = stdy,capsize = 5,ls='None')
        if i == 0:
            locax.plot(lx, theoretical_random(lx,m[j]),
                          ls='--', color=(0, 1 - j / len(m), 1 - i / len(NT)),
                       label= 'Theoretical extended to continuous domain for m = %.0f' %(m[j]))

locax.set_xlim(1,100)
locax.set_yscale('log')
locax.set_xscale('log')
locax.set_xlabel('k')
locax.set_ylabel('Normalized frequencies')
locax.grid()
locax.legend()
plt.show()

