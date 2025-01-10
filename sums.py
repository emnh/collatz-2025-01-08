
import tabulate

from collatz import classical_collatz_cycle_length

def to_bin(x):
    return bin(x)[2:]

maxi = 20
cache = {}
l = []
headers = ("i", "2 ** i", "2 ** (i + 1) - 1", "bin(2 ** i)", "bin(2 ** (i + 1) - 1)", "sum", "sum / range")
for i in range(maxi):
    s = 0
    start = 2 ** i
    end = start * 2
    for N in range(start, end):
        s += classical_collatz_cycle_length(N, cache)
    ratio = s / (end - start)
    l.append((i, start, end, to_bin(start), to_bin(end), s, ratio))
print(tabulate.tabulate(l, headers=headers))