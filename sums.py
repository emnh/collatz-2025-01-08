#!/usr/bin/env python3

import tabulate
import math
from collatz import classical_collatz_cycle_length

def to_bin(x):
    return bin(x)[2:]

def results(start, end):
    clens = []
    for N in range(start, end):
        clen = classical_collatz_cycle_length(N, cache)
        clens.append((clen, N))
    first = lambda x: x[0]
    rsum = sum((first(x) for x in clens))
    rmin = min(clens, key=first)
    rmax = max(clens, key=first)
    rrange = end - start
    rratio = rsum / rrange
    return (rrange, rsum, rratio, rmin, rmax)

maxi = 20
cache = {}
l = []
l2 = []
headers = ("i", "2 ** i", "2 ** (i + 1) - 1", "bin(2 ** i)", "bin(2 ** (i + 1) - 1)", "range", "sum", "sum / range", "min", "max")
maxes = []
for i in range(maxi):
    start = 2 ** i
    end = start * 2
    res = results(start, end)
    l.append((i, start, end, to_bin(start), to_bin(end), *res))
    maxes.append(res[-1])

print(tabulate.tabulate(l, headers=headers))
print("")

ranges = zip(maxes, maxes[1:])
for (start_clen, start), (end_clen, end) in ranges:
    res = results(start, end)
    l2.append((i, start, end, to_bin(start), to_bin(end), *res))
print(tabulate.tabulate(l2, headers=headers))