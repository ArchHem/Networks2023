from network_OOP import *

N_r = 5
N_t = 10000
m = 3
r = 1
scale = 1.2
cutoff = 50
maxx, avgy, avgy_std, maxkavg, maxk_std, fbins = hybrid_system_averager(N_r,N_t,m,r,scale)

y_theo = []
for i in range(len(fbins)-1):

    length = fbins[i+1]-fbins[i]
    indices = np.arange(fbins[i],fbins[i+1],1)
    yl = np.sum(hybrid_distribution(indices,m,r))/length
    y_theo.append(yl)

y_theo = np.array(y_theo)
chi2 = np.sum(((y_theo-avgy)/avgy_std)**2)
chi2 = chi2[maxx < 50]
p = p_value(np.sum(chi2),chi2.shape[0]-3)

plt.plot(maxx,y_theo)
plt.plot(maxx,avgy)
plt.plot(maxx,hybrid_distribution(maxx,m,r))

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel('Normalized frequencies')
plt.show()