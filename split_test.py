import random

def collatz_step(n):
    if n % 2 == 0:
        n //= 2
    else:
        n = 3 * n + 1
    return n

def collatz_step_bit(n):
    if n % 2 == 0:
        n //= 2
        bit = 0
    else:
        on_bits = bin(3 * n)[2:][::-1]
        n = 3 * n + 1
        n_bits = bin(n)[2:][::-1]
        bit = sum([1 if x != y else 0 for x, y in zip(on_bits, n_bits)])
    return (n, bit)

def collatz_cycle_length(n):
    """
    Compute the cycle length of a number n under the Collatz function.
    """
    count = 0
    while n > 1:
        n = collatz_step(n)
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
    maxCount = 10000
    n_orig = N
    aux_orig = aux
    assert isinstance(n, int)
    #print("n", n, type(n))
    clen = collatz_cycle_length(n)
    while n > 1:
        yield (n, aux, divs)
        if n % 2 == 0:
            #assert aux % 2 == 0, aux
            #aux //= 2
            n //= 2
            #n = n // 2
            divs += 1
        else:
            aux *= 3
            n = 3 * n + 1
        count += 1
        #if count >= nlen:
        #print("err maxCount", count, "n", n, "aux", aux, "n_orig", n_orig, "aux_orig", aux_orig, clen)
        #if count >= maxCount:
        #    assert False
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
    maxN = 1000
    while x > 1 and x % 2 == 0:
        n += 1
        x //= 2
        if n >= maxN:
            print("fail rightZeroes", n, x)
            assert False
    return n

def main():
    for N in range(1, 2**8):
        N_bin = bin(N)[2:]
        if N <= 1:
            continue
        N_clen = collatz_cycle_length(N)
        for cutlen in range(2, len(N_bin) - 1):
            left_bin, right_bin = N_bin[:cutlen], N_bin[cutlen:]
            left = int(left_bin, 2)
            right = int(right_bin, 2)
            #print(left_bin, right_bin, N_bin)
            assert len(left_bin) > 1
            assert len(right_bin) > 1
            assert left_bin + right_bin == N_bin
            assert len(left_bin) + len(right_bin) == len(N_bin)
            pad_count = len(right_bin)
            padding = 2 ** pad_count
            left_padded = left * padding
            #zeroes = rightZeroes(left_padded) + 1
            #zeroes = 
            #check = bin(left_padded)[2:][:-zeroes] + right_bin == N_bin
            #if not check:
            #    print(left_padded, bin(left_padded)[2:][:-zeroes], right_bin, N_bin)
            #    assert check
            vals = list(collatz_cycle_length_aux_test_gen(right, left_padded))
            
            last = vals[-1]
            last_aux = last[1]
            counter = 0
            print("len(vals)", len(vals))
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

from collatz import truncate_and_count_ones
from collatz import truncate_and_count_zeroes

def shortcut(n):
    prefix, one_count = truncate_and_count_ones(n)
    cycle_length = 0
    shortcut = ""

    if one_count > 1:
        # Shortcut calculation for ones
        n = 3 ** one_count * (prefix // 2) + (3 ** one_count - 1) // 2
        cycle_length = 2 * one_count + 1
        shortcut = "ones"
    else:
        # Evaluate the zero count of the prefix
        if prefix == 0:  # Avoid infinite loop on zero prefix
            n = 1
            cycle_length += 1
            return n, cycle_length
            
        prefix, prefix_zero_count = truncate_and_count_zeroes(prefix)

        if prefix_zero_count > 1:
            half_zero_count = prefix_zero_count // 2
            n = (3 ** half_zero_count) * prefix
            if prefix_zero_count % 2 == 1:
                n = n * 4 + 1
                shortcut = "zeroOdd"
            else:
                n = n * 2 + 1
                shortcut = "zeroEven"
            cycle_length += 3 * half_zero_count

    return n, cycle_length, shortcut

def reach(n):
    s = 0
    while n > 1:
        n = collatz_step(n)
        nbin = bin(n)[2:]
        s = max(s, len(nbin))
    return s

def pc(inum2, steps, rj, shortfun=lambda x: (x, 0, "")):
    clen = 0
    for i in range(steps):
        bit = 0
        inum2, clen2, shortcutNumber = shortfun(inum2)
        if clen2 > 0:
            clen += clen2
        else:
            inum2, bit = collatz_step_bit(inum2)
            clen += 1
        num2 = bin(inum2)[2:]
        marker = "x"
        if bit > 0:
            num2 = num2[:-bit] + marker + num2[-bit:]
        else:
            num2 += marker
        print("num2", num2.rjust(rj), str(clen).rjust(3), bit, shortcutNumber)
    
def testSome():
    k = 10
    maxLen = 10
    rs = []
    num1 = ''
    for i in range(k):
        r = random.randint(1, maxLen)
        rs.append(r)
        num1 += '0' + '1' * r
    steps = 10
    rj = len(num1) + steps + 10
    print("num1", num1.rjust(rj))
    inum1 = int(num1, 2)
    pc(inum1, steps, rj)
    print("")
    #snum1, _ = shortcut(inum1)
    pc(inum1, steps, rj, shortcut)
    print("")
    for i in reversed(range(len(num1))):
        prefix = num1[:i]
        suffix = num1[i:]
        r = reach(int(suffix, 2))
        #x_mark = ""
        #if r > 0 and r < len(s):
        x_mark = prefix + "x" + suffix
        rs = len(num1) - r
        if r > i:
            rs += 1
        y_mark = x_mark[:rs] + "y" + x_mark[rs:]
        #print(str(r).rjust(4), x_mark.rjust(rj), y_mark.rjust(rj))
        print(str(r).rjust(4), y_mark.rjust(rj))
    # For two numbers a, b where b = collatz_step(a), and the operation is 3n + 1,
    # I want to find the maximum bit of b which is touched by +1.
    # Maybe just compute difference between 3n and 3n + 1?

if __name__ == '__main__':
    testSome()