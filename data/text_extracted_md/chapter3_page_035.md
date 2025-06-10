# Introduction to the Quantum Theory of Solids

However, since the particles are indistinguishable, the \(N_i!\) number of permutations that the particles have among themselves in any given arrangement do not count as separate arrangements. The interchange of any two electrons, for example, does not produce a new arrangement. Therefore, the actual number of independent ways of realizing a distribution of \(N_i\) particles in the \(i\)th level is

\[
W_i = \frac{g_i!}{N_i!(g_i - N_i)!}
\]

(3.77)

## Example 3.5

**Objective:** Determine the possible number of ways of realizing a particular distribution for (a) \(g_i = N_i = 10\) and (b) \(g_i = 10, N_i = 9\).

### Solution

(a) \(g_i = N_i = 10\): We may note that \((g_i - N_i)! = 0! = 1\). Then, from Equation (3.77), we find

\[
\frac{g_i!}{N_i!(g_i - N_i)!} = \frac{10!}{10! \cdot 1} = 1
\]

(b) \(g_i = 10, N_i = 9\): We may note that \((g_i - N_i)! = 1! = 1\). Then, we find

\[
\frac{g_i!}{N_i!(g_i - N_i)!} = \frac{10!}{9!(1!)} = \frac{(10)(9!)}{9!} = 10
\]

**Comment**

In part (a), we have 10 particles to be arranged in 10 quantum states. There is only one possible arrangement. Each quantum state contains one particle. In part (b), we have 9 particles to be arranged in 10 quantum states. There is one empty quantum state, and there are 10 possible positions in which that empty state may occur. Thus, there are 10 possible arrangements for this case.

### Exercise Problem

**Ex3.5** Determine the possible number of ways of realizing a particular distribution if \(g_i = 10\) and \(N_i = 8\). (SPICE)

Equation (3.77) gives the number of independent ways of realizing a distribution of \(N_i\) particles in the \(i\)th level. The total number of ways of arranging \((N_1, N_2, N_3, \ldots, N_n)\) indistinguishable particles among \(n\) energy levels is the product of all distributions, or

\[
W = \prod_{i=1}^{n} \frac{g_i!}{N_i!(g_i - N_i)!}
\]

(3.78)

The parameter \(W\) is the total number of ways in which \(N\) electrons can be arranged in this system, where \(N = \sum_{i=1}^{n} N_i\) is the total number of electrons in the system. We want to find the most probable distribution, which means that we want to find the maximum \(W\). The maximum \(W\) is found by varying \(N_i\) among the \(E_i\) levels, which varies the distribution, but at the same time, we will keep the total number of particles and total energy constant.

We may write the most probable distribution function as

\[
\frac{N(E)}{g(E)} = f_F(E) = \frac{1}{1 + \exp\left(\frac{E - E_F}{kT}\right)}
\]

(3.79)