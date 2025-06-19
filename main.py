import numpy as np

from monto_carlose_price import estimate_price
from Markov_descition_process import mdp


def sortd(datas, labels):

    m = len(datas)
    # print(m)
    datasort = []
    labelsort = []
    mind = []
    labelmin = []
    labelk = 1
    i = 1
    while m >= i:

        mind = min(datas)
        minloc = datas.index(mind)
        lmin = labels[minloc]
        datas[minloc] = 10000

        # print(lmin)

        datasort.append(mind)
        labelsort.append(lmin)
        i = i + 1

    return datasort, labelsort


def cum_sum(d):
    c_sum = 0
    exe = []
    for i in range(len(d)):
        c_sum += d[i]
        exe.append(c_sum)
    return exe


def pres(c_s):
    val = sum(c_s)
    val_e = []
    for i in range(len(c_s)):
        val_e.append((100 * c_s[i]) / val)
    return val_e


def nearest(c, p, label):
    m = []
    for i in range(len(c)):
        m.append(abs(c[i] - p))
    min1 = min(m)
    parat = m.index(min1)

    return label[parat]


def parato(data, label):
    d, l = sortd(data, label)
    c_d = cum_sum(d)

    c_d_pre = pres(c_d)

    val_cum_pre = cum_sum(c_d_pre)

    parato = 80
    print("more likely choose according to parato analysis")

    print(nearest(val_cum_pre, parato, l))


x = []
y = []
z = []
nx = 0
ny = 0
nz = 0
all, x, y, z, nx, ny, nz = estimate_price(1000000)

states = [0, 1, 2]
actions = [0, 1, 2]

# Transition model P[s][a] = [(prob, next_state, reward)]

P = [
    [[nx, 0, 5.1], [ny, 1, 1.8], [nz, 2, 4.8]],
    [[nx / (ny + nx), 1, 3.2], [ny / (nz + ny), 2, 2.6], [nz / (nz + nx), 0, 8.1]],
    [[nx / (nz + nx), 2, 2.1], [ny / (ny + nx), 0, 3.8], [nz / (nz + ny), 1, 6.5]],
]
print(P[1])
v, policy = mdp(states, actions, P)

label = [x, y, z]
score = [v[0], v[1], v[2]]
parato(score, label)
