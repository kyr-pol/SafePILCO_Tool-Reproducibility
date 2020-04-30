import numpy as np

name = '../../../results/X_'

T = 12 * 4
J = 6
N = 5
runs = 5
cost = np.zeros(runs)
violations = np.zeros(runs)
ep_violations = np.zeros(runs)
interactions_blocked = []
for i in range(1,runs):
    X = np.loadtxt( name + str(i) + ".csv", delimiter=',' )

    cost[i-1] = np.sqrt(sum((X[J*T:, 0] - 20)**2 + (X[J*T:, 1] - 20)**2) / (np.shape(X)[0] - T*J) )
    violations[i-1] = X[J*T:][X[J*T:, 1]>20.5].shape[0]

    ep_v = 0
    eps = (X.shape[0] - J*T) // T
    for k in range(0,eps):
        if X[ (J+k)*T : (J+k+1)*T][ X[ (J+k)*T : (J+k+1)*T, 1] >20.5].shape[0] > 0: ep_v += 1
    ep_violations[i-1] = ep_v
    interactions_blocked.append(N - eps)

print(np.mean(cost), np.std(cost))
print(np.mean(violations), np.std(violations))
print(np.mean(ep_violations), np.std(ep_violations))
