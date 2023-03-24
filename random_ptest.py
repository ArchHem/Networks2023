from network_OOP import *

fifr, axr = plt.subplots()
"""Number of realizations"""
N_R = 20
m1 = 3
NT = 20000
p_random = random_p_tester(N_R,NT,m1,axr,a=1.0)
print('Measured P-value against a null hypothesis of a infinite-time distribution is: %.5f' %p_random,
      ' for T = %.0f and m = %.0f'%(NT,m1))
plt.show()