def logical_parity(e, support):
    return sum(e[q] for q in support) % 2
