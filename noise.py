import random

def depolarizing_noise(n, p):
    eX = [0]*n
    eZ = [0]*n
    for i in range(n):
        r = random.random()
        if r < p:
            r2 = random.choice(['X','Y','Z'])
            if r2 in ('X','Y'): eX[i] = 1
            if r2 in ('Z','Y'): eZ[i] = 1
    return eX, eZ
