"""This code makes use of a self-made p-value estimate against the theoretical distribution"""

"""by changing m, we can study different distributions as well"""

from network_OOP import *

N_t = 40000
m = 3
NR = 25
cutoff = 55

sfig, sax = plt.subplots()

p_value = preferential_test(NR,N_t,m,cutoff=cutoff,a=1.2,axplot=sax)

print('For parameters m = %.0f and N = %.0f, averaged over %.0f realizations, '
      'the estimated p-value against the null hypothesis is: %.4f.'
      %(m,N_t,NR,p_value))

plt.show()