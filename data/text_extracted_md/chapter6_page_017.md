# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

## Solution

The condition of a uniform generation rate and a homogeneous semiconductor again implies that \( \partial^2 (\delta p)/\partial x^2 = \delta p(x)/\partial x = 0 \) in Equation (6.56). The equation, for this case, reduces to

\[
g' - \frac{\delta p}{\tau_{p0}} = \frac{d(\delta p)}{dt}
\]

(6.60)

The solution to this differential equation is

\[
\delta p(t) = g' \tau_{p0} (1 - e^{-t/\tau_{p0}})
\]

(6.61)

## Comment

We may note that, as \( t \rightarrow \infty \), a steady-state excess hole and electron concentration of \( g' \tau_{p0} \) is reached. Equation (6.60) contains both a generation rate term and a recombination rate term for the excess carriers.

## Exercise Problem

**Ex 6.3** In Example 6.3, consider n-type silicon at \( T = 300 \, \text{K} \) doped to \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \). Assume that \( g' = 5 \times 10^{21} \, \text{cm}^{-3} \, \text{s}^{-1} \) and let \( \tau_{p0} = 10^{-7} \, \text{s} \). (a) Determine \( \delta p(t) \) at (i) \( t = 0 \), (ii) \( t = 10^{-7} \, \text{s} \), (iii) \( t = 5 \times 10^{-7} \, \text{s} \), and (iv) \( t \rightarrow \infty \). (b) Considering the results of part (a), is the low-injection condition maintained?

\[
\left[ \frac{5 \times 10^{10} \times (1 - e^{-t/\tau_{p0}})}{5 \times 10^{16}} \right] \ll 1 \quad \text{(yes)}
\]

The excess minority carrier hole concentration increases with time with the same time constant \( \tau_{p0} \), which is the excess minority carrier lifetime. The excess carrier concentration reaches a steady-state value as time goes to infinity, even though a steady-state generation of excess electrons and holes exists. This steady-state effect can be seen from Equation (6.60) by setting \( d(\delta p)/dt = 0 \). The remaining terms simply state that, in steady state, the generation rate is equal to the recombination rate.

## Test Your Understanding

**TYU 6.1** Silicon at \( T = 300 \, \text{K} \) has been doped with boron atoms to a concentration of \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). Excess carriers have been generated in the uniformly doped material to a concentration of \( 10^{15} \, \text{cm}^{-3} \). The minority carrier lifetime is 5 μs.  
(a) What carrier type is the minority carrier?  
(b) Assuming \( g' = E = 0 \) for \( t > 0 \), determine the minority carrier concentration for \( t > 0 \).

\[
\left[ \frac{10^{15} \times e^{-t/\tau_{p0}}}{5 \times 10^{16}} \right] \ll 1 \quad \text{(yes)}
\]

**TYU 6.2** Consider silicon with the same parameters as given in TYU 6.1. The material is in thermal equilibrium for \( t < 0 \). At \( t = 0 \), a source generates excess carriers turned on, producing a generation rate of \( g' = 10^{20} \, \text{cm}^{-3} \, \text{s}^{-1} \).  
(a) What carrier type is the minority carrier?  
(b) Determine the minority carrier concentration for \( t > 0 \).  
(c) What is the minority carrier concentration as \( t \rightarrow \infty \)?

\[
\left[ \frac{10^{20} \times \tau_{p0} \times (1 - e^{-t/\tau_{p0}})}{5 \times 10^{16}} \right] \ll 1 \quad \text{(yes)}
\]