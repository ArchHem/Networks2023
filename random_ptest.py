from network_OOP import *

fifr, axr = plt.subplots()
"""Number of realizations"""
N_R = 20
N1 = 10000
E1 = 40000
p_random = random_p_tester(N_R,N1,E1,axr,a=1.0)
print('Measured P-value against a null hypothesis of a binomial distribution is: %.5f' %p_random,
      ' for N = %.0f and E = %.0f'%(N1,E1))
plt.show()