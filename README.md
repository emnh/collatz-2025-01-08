# Collatz Experiments

## Resources

- [Wikipedia on Collatz Conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)
- [Convergence verifcation of the Collatz problem, David Barina](https://www.fit.vut.cz/research/publication-file/12315/postprint.pdf)
- [David Barina's Code](https://github.com/xbarin02/collatz/)
- [GPU-accelerated Exhaustive Verification of the Collatz Conjecture: Takumi Honda, Yasuaki Ito, and Koji Nakano](http://www.ijnc.org/index.php/ijnc/article/view/135/144)
- [Eric Rosendaal's Page of Collatz Records](http://www.ericr.nl/wondrous/)
- [Almost all orbits of an analogue of the Collatz map on the reals attain bounded values](https://arxiv.org/abs/2401.17241)
- [An Automated Approach to the Collatz Conjecture](https://arxiv.org/abs/2105.14697)

http://www.ericr.nl/wondrous/
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

# Other ideas from my todoist

The general idea is to get creative brainstorming and attack Collatz from as
many perspectives as possible to build out the intuition around the problem
to try to grasp why the Conjecture is true or false and then if Eureka hits,
try to condense it down to a mathematical proof of the intuitive idea. It
doesn't work with all problems, because the abstract cannot be gained easily
from the concrete, but I think this problem fits nicely for the approach of
building out the intuition bottom-up, rather than sitting and playing with
deductions from a more abstract perspective to understand. The latter is,
of course, also a valid approach, and probably the chosen path of many, but
it's not my favourite attack vector. I like to play with the concrete. Then
when I get tired of it, I might lazily try to eliminate my own work on the
concrete by abstracting it with formulae.

## Collatz subsequence linear transformations validation

Generate all binary possibilities of Collatz subsequences and narrow down the
valid ones by calculating the linear transformations and locating them in the
integers by sliding to the applicable place by monotonicity of the transformation,
check for valid integers that the subsequence takes from one to another, and then
for infinitely many insertion points potentially if there is some multiplicative
generalization or other. Then try to group up sets of subsequences in the generation
such that we concern ourselves with the valid ones only, and then check if this
maps out the terrain either in a complete way as towards a proof or at least
intuitively builds more understanding of the Collatz space to contribute towards
other ideas. I don't think it's going to lead to a proof, but I might learn
something.

## Check what happens with Collatz if adding 1(0*) to the left of a binary number.

## Natural Numbers Game

https://adam.math.hhu.de/#/g/leanprover-community/NNG4

## Collatz implementation trivial ideas

Write trivial programs for addition, subtraction (?), multiplication and
division. Start with base 10, then generalize to any base and then to any digit
set for the base multipliers. See if this has any application to 3n + 1 when
done, by checking if you can enumerate Collatz in some way by shuffling which
operations are performed in sequence. Try to apply Collatz to constituents of a
numeric decomposition of a number and see whether it's possible to decompose the
Collatz sequence itself, mostly at current point to build a better intuition of
what goes on with Collatz sequences.

## Investigate Collatz Conjecture aka 3n + 1 problem

## Write down lots of 3n+1 tasks under the relevant section, including completed, incomplete and wild ideas.

## Read [The Collatz conjecture, Littlewood-Offord theory, and powers of 2 and 3](https://terrytao.wordpress.com/2011/08/25/the-collatz-conjecture-littlewood-offord-theory-and-powers-of-2-and-3/)

## Who said something along the lines of that when you get your data structures in order, the algorithms kind of fall into place? ChatGPT says Donald Knuth. I guess this applies to 3n+1 also, in that number representations are data structures.

## Investigate filling the space of all integers with reverse Collatz.

## Investigate the idea of factoring a number N to p, a set of primes, times an exponent of 2 times an exponent of 3. Also know that the exponent of 2 should immediately be deleted. Investigate writing p to be a base 2 or base 3 number or a sum, for example 5 equals 2 plus 3.

## Try any repeated binary pattern of zeroes and ones (1001001010101) repeated n times and see if you can generalize what happens to it with the Collatz sequence. Ask ChatGPT first.

## Try to make some small random algebraic abstractions over 3n + 1 and import some symbolic algebra package in python3 and play with it. Machine learning could be useful to build the AIs neural network intuition for the correspondence between number classes and formulae, but I think it's too much technological overhead to motivate me. Perhaps investigate if there are finished solutions for it already that need less work to adapt.

## If successfully conquering repetitive binary sequences in Collatz then try non-repetitive sequences, for example irrational numbers like sqrt(2) and for example Pi or other transcendental numbers and check what happens if you feed progressively longer subsequences of those to 3n + 1.

## 

## Collatz Conjecture

## A power of 2 can be written like the sum of two numbers in binary in any way that the "mente" from the two rightmost ones add up and flip the bits to the left of that to 0 (so that the digits in each index i of the two numbers must be inversions, thus giving the 1-ladder that the rightmost equal digit can flip.

## You might try to reason about the number of ones, and similarly the distance between the leftmost and the rightmost one-digit, in the binary number expansion of N in the Collatz conjecture.

## Maybe try to write 3n like 2n + n and then 3n + 1 like 2n + n + 1 and think about the previous statement in that regard, writing the number N and thereby 3N + 1 like the summation formulae abstracted over any number.

## You can ask the equivalent (?) question: Can any number be written to be an expansion of 2^n by the rules above (but increasing rigor if it is indeed a possible representation)?

## If the Collatz conjecture is true, every power of 2 should be expandable by reversing Collatz operations to any number below a certain bound? Or perhaps multiple power of 2 are required. Ahh yes I think probably so.

## Write a program that keeps track of "divisions by two" or equivalently "removing zeroes on the right hand side in the binary expansion" and look at which numbers "land" on each power of two, disregarding the "undeleted zeroes" on the right hand side.

## In the opposite you can multiply by two until the number minus one is divisible by 3.

## Download a whole lot of integer sequences from [The On-Line Encyclopedia of Integer Sequences® (OEIS®)](https://oeis.org/) and see whether they have interesting properties when fed to Collatz operations. Especially if any of them have a relatively high cycle length compared to the input number, to see if there is a chance of creating an infinite sequence or a cycle.

https://oeis.org/stripped.gz

## What happens if you feed the cycle length of a cycle recursively back into Collatz?

## Another fun experiment could be to teach a neural network some connection between computer programs and OEIS integer sequences, but more technogical integration work that might be beyond my tech motivation.

## Maybe read some articles from Jeffrey Lagaris, quoted to be expert on 3n + 1 by Veritasium. Skim the language but skip most mathematical formulae due to parsing cost of mathematical formulae against the network.

[https://dept.math.lsa.umich.edu/~lagarias/3x+1.html](https://dept.math.lsa.umic
h.edu/~lagarias/3x+1.html)  Introduction maybe? [https://maa.org/sites/default/f
iles/pdf/upload_library/22/Ford/Lagarias3-
23.pdf](https://maa.org/sites/default/files/pdf/upload_library/22/Ford/Lagarias3
-23.pdf)  ChatGPT also suggested a survey or cataloguing article by him:
[\[math/0608208\] The 3x+1 Problem: An Annotated Bibliography, II (2000-
2009)](https://arxiv.org/abs/math/0608208#:~:text=The%203x%2B1%20problem%20conce
rns,n%20includes%20the%20integer%201).

## Terence Tao results

[The Collatz conjecture, Littlewood-Offord theory, and powers of 2 and
3](https://terrytao.wordpress.com/2011/08/25/the-collatz-conjecture-littlewood-
offord-theory-and-powers-of-2-and-3/)  [Almost all Collatz orbits attain almost
bounded values](https://terrytao.wordpress.com/2019/09/10/almost-all-collatz-
orbits-attain-almost-bounded-values/)

## After feeding the OEIS into Collatz, try to relate automatically sequences that are always bigger or smaller in cycle length or mixed relation to other sequences, some growth comparison perhaps. Then you can map out the space of a lot of known sequences to check their interactions and make a graph of it.

## Try to check OEIS for sequences that have a constant cycle increase or arithmetic series cycle increase, such that the formula for the sequence can be used to generate cycles with a desired number. Remember that strings of binary 1000... and 1111... already have shortcut expansions.

## Try to divide and conquer the problem.

Write the number to be a sum a + b where a is concerned with the most
significant digits of the problem and b is concerned with the least significant
digits of the problem. Determine when there is an interaction from b overflowing
into a and when this interaction will change the course of which operations are
going to be performed on the number, that is, how many operations can be made on
part b before consuming part a. Assume that all number lower than some limit N
have been proven and you could also try induction then. If all numbers greater
than N can be decomposed into parts lower than N by the algorithm and combined
with an algorithm that has an upper bounded number of steps then we are done.

## Play more with base 2 and 3 representations, also in combination with a + b divide and conquer.

## Chain gang

I'm looking for any algorithm which provably takes a finite number of steps to
calculate the Collatz cycle length, bounded above by some constant, or bounded
by any other algorithm that takes the first algorithm and computes the number of
steps that the first takes in a finite number of steps. So in this way one can
actually build a chain of k algorithms, where k is finite and each meta-level
algorithm that takes another algorithm as input and determines in a fewer, or
actually any finite number of steps that could actually be higher also but
calculatable to a finite number that is easy to prove the upper bound for.

## Infinite ways of constructing numbers

Try writing the algorithm that computes in a structured way perhaps all cycle
lengths for numbers less than N and prove that this algorithm terminates. You
can try with concatenation of digits, replacement of digits, different bases,
sums, multiplications, prime factor construction of all numbers less than N and
prove that the algorithm terminates in a bounded number of steps depending on N.

## The fastest growing hypothetical and infinite sequence is just an infinite loop starting with any N and just applying 3n + 1. Then we can divide every number in this sequence by 2 an infinite number of times. Now we have a two dimensional infinite grid that we can try to find the substructure of and see if we can bound the size of it depending on N.

## Break all involved algorithms into basic algorithms for addition, long multiplication, long division and so on and try to reorder the steps of the algorithm in a way that makes it easier to analyze and bound the number of steps it takes.

## For a given Collatz operation sequence, find the upper bound of numbers that shrink and the lower bound of numbers that grow given this sequence, potentially using binary search. Filter by valid integer results if needed.

## Three tasks (subsequence search, integer test for sequence compat, reconstruct)

Three tasks: Implement binary search to look for numbers compatible with a
subsequence of Collatz operations. That means a sequence of operations that is
either the start, middle or end of another sequence, which means it takes some
integer a to the integer b without either a being the start of the sequence
necessarily (but it could be) and b not necessarily being the end, then usually
1 in the common case. Run the sequence backwards from 1 or any number and check
if that's compatible. Try to reconstruct any number based on a sequence of
Collatz operations (wait, isn't that trivial, just run it backwards with inverse
operations, I already have that function?).

## Function bounding

For any function f that is a composition of a sequence of Collatz operations,
try to find the range (probably one integer, the closest one) where the function
f diverges on either side or converges in the case of a cycle. If it takes
numbers down to one that's the classical sequence you get from running the
operations conditional on a particular number. Then how do I know which number,
well just test it in reverse from 1 to check if it's such a sequence. Let's
specify more formally. For any integer x and any function f, then let b = f(a)
and c = f(f(a)), then one can, presumably, determine the smallest such integer b
and corresponding a or a and corresponding b (probably that will be the same
pair yes?) and if c == a then it's a cycle. We can instead write g(c) =
f^(-1)(c) = f(a) because then presumable we have two searches that have to meet
in the middle for it to be a cycle. Then we can for any function f and its
reverse g test whether it leads to a cycle if we can show that f and g meets
somewhere or diverge based on both functions being monotonically increasing or
decreasing with respect to being bigger or smaller than a particular number x.
After we have shown that to be feasible using computer code then we can check if
we can abstract over all functions of Collatz sequences f because they are
forming ordered relations with subsequences of Collatz. Either that or actually
I would first try to start a search that based on a sequence that doesn't have a
cycle, whether it can create another sequence that comes closer to creating a
cycle. There is a question here that computer code can answer whether and how to
demonstrate these monotonically increasing and decreasing ranges of the
functions. First tests are just to test inefficiently for small ranges that the
assumptions are valid, then make it more efficient by reintroducing the binary
search written before and correctly based on the verified assumptions about
monotonicity so we can efficiently locate the ranges of each function and
whether they can meet somewhere in the middle. I'm already too vague on the
numerical intuitions about growing and shrinking functions so I need to refresh
my intuition with results of running code or symbolic manipulations perhaps but
that would be more costly I think. Better to first verify with some numbers and
then try to translate that into formulae which can be logically verified for all
numbers. There was something about squeezing a function h between f(x) and x if
f(x) < x or f(x) > x such that h is closer to x, but now we have extended it. So
if f takes a to b and g takes b to a then we have a cycle so any function that
goes upward must have a reverse that takes it back again. And then the check for
a cycle if is fgfg(x) = x. Why do I need to decompose into one function that is
increasing and one decreasing? Can't I just assume that we can search directly
for the composition fg and then check if for every number greater than x it
diverges upward and similarly for any number lower than x it diverges downward?
Why is it advantageous to decompose and have two searches? For showing
impossibility of them meeting intuitively or what? Can't I just search for any
full cycle fg and show that it diverges either upward or downward, and to look
for a better alternative function one tries to look for functions that diverge
more slowly and show that there is a limit where you may see them diverge so
slowly that it forms a cycle? I think it's much easier or faster to show that or
look for f and g where g is easy to produce from f, but can I show that any
cycle can be written fg? Well actually fg(x) = x by definition so the previous
test is not for whether fgfg(x) = x, that's by definition, it's whether f and g
are indeed both valid for any integers, which hopefully we can test by using
properties of monotonicity to restrict the search. Let it be said that f is a
sequence of 1s and 0s where 1 means take x to 3x + 1 and 0 means take x from x
to x / 2 and the validity checks are that x must be odd at the time there is a 1
and x must be even at the time there is a 0 for the function string to be valid.
There is one caveat, even if we can prove that there are no cycles, we still
have to show that there is no divergent sequence that goes upward to infinity.
What I can say is that there can not be any repetition in the function string f
if we prove that there is no cycle in the function (other than the 4,2,1 cycle
of course), so if there is no repetition I feel something vague about the state
space of the function always growing if it goes to infinity. Well that's already
known given that it grows to infinity. If it goes to infinity then f(x) > x, not
necessarily for every x, but there exists some x for which f(x) > x and any
repeated application of f to x is greater than one less repeated application of
f to x. To try to understand the problem better intuitively try to create an f
that always increases. We know that any number in binary plus a repeated string
of 1s already goes quite high based on the 1s being repetitively and easily
reduced to multiplication by 3 to the power of n, some function of the number of
1s (look at earlier code for specifics), so we can start with that one and look
at why it doesn't go to infinity. Then if we understand that we can try to
dismiss other functions also.  So in summary what I have done is that for every
application of the Collatz algorithm I say that it produces an output string of
1s for application of the first rule and 0 for the application of the second
rule, by convention, and then I consider the space of all those possible and
impossible (let's say invalid) outputs, checking first for the cycle property by
a potentially binary search based on monotonicity and secondly for the validity
property. If I can dismiss all based on the cycle property not being met I'm
good but if it secondly depends on the validity property then it's still
dependent on the actual numbers so the trick didn't work then if it doesn't
remove that dependency. If I do find some invalid cycles I can still try to
generalize and figure out why they are not valid in general over all such
cycles.

## Differentiating monotonic functions

I I feel particularly inclined for doing the differentiations and want to avoid
coding intuitively I can try to differentiate f and g from the previous TODO
item about "Function bounding" and see where they intersect. It may be not too
difficult after all and if I plug into a python3 library for algebraic
differentiation or do at least some tests using Wolfram Alpha (if it generalizes
that's fine, if it depends on a specific sequence f of functions how it pans out
then I should look at the output of each function being differentiated by some
symbolic manipulation library and see if that looks intuitively easier to
generalize). Actually it can be interesting to see what longer compositions of
0s and 1s, that is, the Collatz operations, look like algebraically when
simplified.

## Infinity sums

If you sum up or sum up the upper bound of steps that some algorithm that runs
over all numbers from 1 to a function f(N) when computing Collatz, what happens?
And what will be the difference of the growth of this sum if there is one number
that the Collatz function does not terminate for and if it does terminate for
every number? Try to find another function different from the Collatz that does
have a number of infinite cycles where you can investigate the difference
intuitively based on numerical calculations, perhaps. Decide whether to run any
Collatz function of a number x for a finite number of cycles or let it run
forever until 1. Clearly, if the Collatz conjecture is true then the sum of the
end number of Collatz(x) for x from 1 to N is N, and infinite if Collatz
conjecture is not true. Try maybe to introduce the idea of a limit on size of
the state space of the algorithm that computes Collatz.

## There can be 2 dimensions to grow, the starting number N and the number of steps each Collatz application is allowed to run. Try to make a correlation between the two.

## If there is no cycle in the Collatz output, that means that there is no number N where for N / 2 the first half of the Collatz output sequence is equal to the second half.

## If there is an infinite sequence generated by Collatz, how do you prove that it is infinite? Well, you must prove that for every x, there is some number of applications of Collatz f where f^n(x) > x.

## If there is an infinite Collatz application that takes a number upwards to infinity, then for every x >= s there exists a number n such that c^n(x) > x, otherwise the state space of the algorithm is limited and it must either cycle in that state space or terminate in 1 (why in 1? probably because of monotone decreasing function). If there is no cycle there is no subsequence at the end of the Collatz sequence which can be repeated and lead to the same number (why?).

## Treat the game of Collatz like a whackamole. If there is a number of criteria or numbers that grow based on calculation of Collatz for any number from 1 to N, then if you can enumerate and bound all of the whackamole scenarios then you have confined the Collatz applications to a finite space. Now show that no infinite cycle or sequence can escape your bounds.

## Bounded space

For any bounded space, say limiting the number of steps Collatz is allowed to
run and the size of the space of the input numbers and the size of the space of
the state space of the program and the maximum number that Collatz is allowed to
reach, figure out what does it mean for this bounded space if Collatz is true,
false and if it is cycling or infinite. Assume you have an infinite growth,
apply to the growing bounded space, and assume you have a finite cycle and apply
to the growth of the bounded space, see what happens.

## State space

A non-terminating algorithm is an algorithm for which, if you add to it the
capability to remember every state it has been in and terminate if it reaches
the same state twice, then for every set of its state space, there exists at
least one state in the state space for which the algorithm increases the size of
the state, otherwise there would be an upper bound on the size of the state
space and then there must be a cycle in the state space terminating the
algorithm. Why? Well, for every sequence of states the program goes through,
there is either an upper bound on the size of the longest state and then there
is a finite number of states shorter or equal to that size, which means that any
infinite sequence of states is repeating a state if it is supposed to map finite
to infinite (infinite is bigger than finite, so that means that by the
pigeonhole principle there is one state with more than one occurence).

## Terminating algorithms

You have one algorithm A that you don't know if it terminates. Based on A you
construct another algorithm B which you know terminates and then you prove that
if B terminates then A terminates. Perhaps B is a checksum of the output of A,
perhaps B is in another way an upper bound of the steps that A can take. It is
another take on the halting problem.

## Completion

To complete the state space, just add 1 for every even number and subtract 1 for
every odd number. Now examine the state space.

## Checking for cycles

For every bit pattern, if it is a Collatz cycle, then any rotation of the bit
pattern is also a cycle, starting at another integer with the cycle. This can
probably be used to exclude a lot of patterns fast. For an N-bit pattern, there
are at most N rotations, while some patterns like only 1s or only 0s only have 1
rotation. (10)* only has two rotations. Look into cyclic groups maybe to check
in general.

## Relation to cyclic groups

Is it true that a number of rotations N of a binary sequence of operations that
correspond to linear transformations corresponds to a cyclical group of length
N? How, given a binary sequence does one determine which starting integer it
would be cyclical for, if there is such an integer?

## Cyclic groups

Any cycle of integers under the Collatz operations maps to a string of 0s and 1s
for the two operations and the number k from to 1 to N, the length of the
string, can be viewed as a number of rotations of the string under the cyclic
group Z / nZ. What happens now if we apply the Collatz operations to the cyclic
group or otherwise determine its correspondence to this cyclic group? Well, we
have a sequence of integers which correspond to a cyclic group but is generated
by a recurrence relation. Figure out the connection. If our cycle can be
factored into prime length subcycles, then the isomorphic structure of integers
should be factorizable in the same way, no? Also figure out what happens to the
linear transformation ax + b under rotation of binary strings of operations,
what is the range of the parameters a and b and how to find the rotation that
maximizes or minimizes a or b?
https://chatgpt.com/c/5d980174-80e9-4ce5-86b4-820d39591f9b

## For a valid Collatz sequence, check its linear transformations. Generate these linear transformation sequences for numbers from 1 to N. Also check how to generate longer Collatz sequences from a set of binary strings and check the growth of the size of the set and whether it can be easily calculated.

## I would like to know also, for k 1-operations and l 0-operations, what is the closest integer to x such that f(x) = x.

## For every f(x) = ax + b, I can generate the inverse r(x) = x = (f(x) - b) / a by reversing and inverting the operations, and then try to figure a forward sequence g(x) that results in the same linear transformation as r(x) such that f(g(x)) = x.

## The n = ax + b transformation has a = 3^n/2^k where n is the number of ones in the digit string and k is the number of zeroes. The b has a more complex expression it looks like. For any binary string that starts with 1, b grows faster than a, given that for every one it both multiplies by 3 and adds 1 while a is just multiplied by 3 and both divide by 2 for each zero.

## Read someone's attempted proof

https://arxiv.org/pdf/2402.00001

## Interrogate PDF on Collatz

[AskYourPDF: The Best PDF AI Chat App](https://askyourpdf.com/conversations/c/84
8717fb-3b22-4a63-91fc-a5c41f9fa775)

## Get inspired by cobweb plots and try to make one.

[Cobweb plot - Wikipedia](https://en.wikipedia.org/wiki/Cobweb_plot)

## Read a bit about FRACTRAN

[FRACTRAN - Wikipedia](https://en.wikipedia.org/wiki/FRACTRAN)

## Simple Programming Tasks

Try to learn about the numerical structure by trying to optimize calculations
of Collatz over ranges. You may try sum, average, min, max and whatever else
you come up with.

## Explore various CFG string generation schemes

List some productions to generate binary strings and check how it interacts with Collatz.
