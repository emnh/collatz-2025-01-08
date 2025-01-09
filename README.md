# Collatz Experiments

## Check out how Collatz can be short-circuited and perhaps bounded when entangled

Divide a binary string into left concat right. And perhaps some playing space
made of "0"s in between. Check whether we can, or for how many iterations we
can run Collatz on the right part without disturbing the left part other than
multiplication by 3 and division by 2. In other words, check that the 3n and
+1 operations do not overflow into the left part with the carry. Then cache
all the applications to the right part including how much space they will need
to expand by overflow for each application (say 4 zeroes in the middle gives
you a lot more ways to run Collatz on a right part, because the overflow
buffer is bigger). Check under which circumstances entanglement happens fast
so that there isn't much to gain with a shortcut. Compare such entanglements
relatively against other entanglements which attempt to maximize the Collatz
cycle length, so that one can relatively bound them, or aternatively look for
the maximum Collatz cycle length for k bits using these techniques.
