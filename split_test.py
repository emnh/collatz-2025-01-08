import random

def collatz_cycle_length(n):
    """
    Compute the cycle length of a number n under the Collatz function.
    """
    count = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        count += 1
    return count

def collatz_cycle_length_aux_gen(n, aux):
    """
    Compute the cycle length of a number n under the Collatz function.
    """
    count = 0
    divs = 0
    while n != 1:
        yield (n, aux, divs)
        if n % 2 == 0:
            assert aux % 2 == 0
            aux //= 2
            n //= 2
            divs += 1
        else:
            aux *= 3
            n = 3 * n + 1
        count += 1
    yield (n, aux, divs)

def collatz_cycle_length_aux_test_gen(n, aux):
    """
    Compute the cycle length of a number n under the Collatz function.
    """
    count = 0
    divs = 0
    while n != 1:
        yield (n, aux, divs)
        if n % 2 == 0:
            #assert aux % 2 == 0
            aux //= 2
            n //= 2
            divs += 1
        else:
            aux *= 3
            n = 3 * n + 1
        count += 1
    yield (n, aux, divs)


def collatz_cycle_length_gen(n):
    """
    Compute the cycle length of a number n under the Collatz function.
    """
    count = 0
    while n != 1:
        yield n
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        count += 1
    yield n

def random_binary_string(k):
    """
    Generate a random binary string of length k.
    """
    return ''.join(random.choice('01') for _ in range(k))

def isEven(n):
    return n % 2

def cycleEven(ns):
    return [isEven(x) for x in ns]

def compute_collatz_lengths(l, k):
    """
    For a random k-digit binary string b, compute the cycle length of all binary strings from 1 to 2^l
    concatenated with "00" and b.

    Parameters:
    l (int): Number of binary strings to iterate over (from 1 to 2^l).
    k (int): Length of the random binary string b.

    Returns:
    dict: A dictionary where keys are binary strings of length l and values are their Collatz cycle lengths
    after concatenation with "00" and b.
    """
    b = random_binary_string(k)
    b = "1" * k
    print("b" + b, [bin(x) for x in list(collatz_cycle_length_gen(int(b, 2)))])
    results = {}

    for i in range(0, 2**l):
        # Convert i to a binary string of length l
        a = f"{i:b}".zfill(l)
        # Concatenate a, "00", and b to form the full binary string
        full_binary = a + "00" + b
        # Convert the full binary string to an integer
        n = int(full_binary, 2)
        ns = list(collatz_cycle_length_gen(n))
        cycles = cycleEven(ns)
        ns2 = zip(ns, cycles)
        a_num = int(a, 2)
        b_num = a_num
        div = 0
        mul = 0
        for even in cycles:
            if even:
                div += 1
                #b_num //= 2
            else:
                mul += 1
                #b_num = 3 * b_num
        if b_num > 0:
            l1 = collatz_cycle_length(b_num)
            l2 = l1 * (3 ** mul)
            #l2 = l1 * (3 ** mul) // (2 ** div)
            if l2 > 0:
                l2 = collatz_cycle_length(l2)
            else:
                l2 = 0
            #assert(l2)
            print("L", l1, l2)
            #n_len = collatz_cycle_length(n)
            #b_num_clen = collatz_cycle_length(b_num)
            #print("N", n, n_len, "A", a, a_num, "B", b_num, b_num_clen, "C", len(cycles))
        #print(full_binary, "\n".join([(bin(x)[2:] + ":" + bin(y)[2:]) for x, y in ns2]))
        #print("")
        # Compute the Collatz cycle length of n
        results[a] = collatz_cycle_length(n)
    print("")

    return results

def split_test(a, b):
    #a = random_binary_string(k)
    #b = random_binary_string(k)
    na = int(a, 2)
    nb = int(b, 2)
    nmid = "00"
    full = a + nmid + b
    nfull = int(full, 2)
    if na > 0 and nb > 0 and nfull > 0:
        #na_clen = collatz_cycle_length_aux_gen(na, a)
        #nb_aux = list(collatz_cycle_length_aux_gen(nb, na))
        #nfull_clen = collatz_cycle_length(nfull)
        a_mul = 2 ** (len(full) - len(a))
        a_trans = na * a_mul
        nfull_aux = collatz_cycle_length_aux_gen(nfull, a_trans)
        #ns1 = list(collatz_cycle_length_aux_gen(nb, a))
        #print("A", a, na, na_clen)
        #alternate = collatz_cycle_length(na)
        #na_last = nb_aux[-1][1]
        check = True
        for val, aux, divs in nfull_aux:
            #if divs > 0:
            #    print("AUX", aux, divs, aux // divs)
            val_bin = bin(val)[2:]
            assert aux % a_mul == 0
            aux_bin = bin(aux // a_mul)[2:]
            prefix = aux_bin
            check = val_bin.startswith(prefix)
            check_count = 0
            for x, y in zip(val_bin, prefix):
                if x == y:
                    check_count += 1
                else:
                    break
            check_total = str(min(len(val_bin), len(prefix)))
            #check2 = val_bin.startswith(b)
            #na_clen = collatz_cycle_length(na * aux * 4)
            #while divs > 0 and na_clen % 2 == 0:
            #    divs -= 1
            #    na_clen = na_clen // 2
            print("V", check, "C", str(check_count) + "/" + check_total, val, aux, divs)
        return check
            #print("F", nfull_clen, "L", len(nb_aux), "NA", na_clen, "B", val, aux, divs)
        #na_clen = collatz_cycle_length(nb_aux[-1][1])
        #nb_clen = len(nb_aux)
        #print("B", b, nb, nb_clen, nfull_clen, na_clen)
        #print("F", full, nfull, nfull_clen)
        #joint = []
        #for x, y in nb_clen:
        #    joint.append((x, y))
        #print("R", *joint)
    else:
        print("zero", a, b, full)
        return False

def test_all(n):
    checks = {}
    no_checks = {}
    for an in range(1, n + 1):
        for bn in range(1, n + 1):
            a = bin(an)[2:]
            b = bin(bn)[2:]
            print("A", a, "B", b)
            check = split_test(a, b)
            t = (a, b)
            if check:
                if not t in checks:
                    checks[t] = 1
                else:
                    checks[t] += 1
            else:
                if not t in no_checks:
                    no_checks[t] = 1
                else:
                    no_checks[t] += 1
            print("")
    for k, v in checks.items():
        print("Check", k, v)
    for k, v in no_checks.items():
        print("No check", k, v)
#test_all(2**4)

def rightZeroes(x):
    n = 0
    while x > 1 and x % 2 == 0:
        n += 1
        x //= 2
    return n

for N in range(1, 2**8):
    N_bin = bin(N)[2:]
    if N <= 1:
        continue
    N_clen = collatz_cycle_length(N)
    for cutlen in range(1, len(N_bin)):
        left_bin, right_bin = N_bin[:cutlen], N_bin[cutlen:]
        left = int(left_bin, 2)
        right = int(right_bin, 2)
        #print(left_bin, right_bin, N_bin)
        assert left_bin + right_bin == N_bin
        pad_count = len(right_bin)
        padding = 2 ** pad_count
        left_padded = left * padding
        vals = list(collatz_cycle_length_aux_test_gen(right, left_padded))
        
        last = vals[-1]
        last_aux = last[1]
        counter = 0
        for val in vals:
            val_N, val_aux, val_divs = val
            val_divs_2 = 2 ** val_divs
            if False and val_divs_2 > 0:
                print("vals", val_aux, val_divs_2)
                assert val_aux % val_divs_2 == 0
                val_aux = val_aux // val_divs_2
            #val_aux_padded = val_aux * padding
            inner_left, inner_right = val_aux, val_N
            left_len = len(bin(inner_left)[2:])
            right_len = len(bin(inner_right)[2:])
            zeroes_on_left_right = rightZeroes(inner_left)
            if zeroes_on_left_right >= right_len:
                counter += 1
            #if val_N:
            #if N_clen == 
            #print(val)
        #val_N, val_aux, val_divs = last
        print("N", N, "N_clen", N_clen, "left_bin", left_bin, "right_bin", right_bin, counter, counter)
    print("")
        #print("aux", aux, "N", N)