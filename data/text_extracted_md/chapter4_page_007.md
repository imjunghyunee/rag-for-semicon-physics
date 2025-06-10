# Chapter 4: The Semiconductor in Equilibrium

Equation (4.14) may be solved more easily by again making a change of variable. If we let

\[
\eta' = \frac{E_v - E}{kT}
\]

then Equation (4.14) becomes

\[
p_0 = \frac{4\pi (2m_p^* kT)^{3/2}}{h^3} \exp \left( \frac{-(E_F - E_v)}{kT} \right) \int_0^\infty (\eta')^{1/2} \exp(-\eta') \, d\eta'
\]

where the negative sign comes from the differential \(dE = -kT d\eta'\). Note that the lower limit of \(\eta'\) becomes \(+\infty\) when \(E = -\infty\). If we change the order of integration, we introduce another minus sign. From Equation (4.8), Equation (4.16) becomes

\[
p_0 = 2 \left( \frac{2\pi m_p^* kT}{h^2} \right)^{3/2} \exp \left( \frac{-(E_F - E_v)}{kT} \right)
\]

We may define a parameter \(N_v\) as

\[
N_v = 2 \left( \frac{2\pi m_p^* kT}{h^2} \right)^{3/2}
\]

which is called the **effective density of states function in the valence band**. The parameter \(m_p^*\) is the density of states effective mass of the hole. The thermal-equilibrium concentration of holes in the valence band may now be written as

\[
p_0 = N_v \exp \left( \frac{-(E_F - E_v)}{kT} \right)
\]

The magnitude of \(N_v\) is also on the order of \(10^{19} \, \text{cm}^{-3}\) at \(T = 300 \, \text{K}\) for most semiconductors.

----

## Example 4.2

**Objective:** Calculate the thermal-equilibrium hole concentration in silicon at \(T = 400 \, \text{K}\).

Assume that the Fermi energy is 0.27 eV above the valence-band energy. The value of \(N_v\) for silicon at \(T = 300 \, \text{K}\) is \(N_v = 1.04 \times 10^{19} \, \text{cm}^{-3}\). (See Appendix B)

### Solution

The parameter values at \(T = 400 \, \text{K}\) are found as:

\[
N_v = (1.04 \times 10^{19}) \left( \frac{400}{300} \right)^{3/2} = 1.60 \times 10^{19} \, \text{cm}^{-3}
\]

and

\[
kT = (0.0259) \left( \frac{400}{300} \right) = 0.03453 \, \text{eV}
\]

The hole concentration is then

\[
p_0 = N_v \exp \left[ \frac{-(E_F - E_v)}{kT} \right] = (1.60 \times 10^{19}) \exp \left( \frac{-0.27}{0.03453} \right)
\]

or

\[
p_0 = 6.43 \times 10^5 \, \text{cm}^{-3}
\]