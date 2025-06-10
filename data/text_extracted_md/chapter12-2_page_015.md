## 12.4 Nonideal Effects

!Graphs

**Figure 12.23** | Minority and majority carrier concentrations in the base under low and high injection (solid line: low injection; dashed line: high injection).

**Figure 12.24** | Common-emitter current gain versus collector current. *(From Shur [14].)*

Two effects occur in the transistor at high injection. The first effect is a reduction in emitter injection efficiency. Since the majority carrier hole concentration at \( x = 0 \) increases with high injection, more holes are injected back into the emitter because of the forward-biased B–E voltage. An increase in the hole injection causes an increase in the \( J_{cE} \) current and an increase in \( J_{pE} \) reduces the emitter injection efficiency. The common-emitter current gain decreases, then, with high injection. Figure 12.24 shows a typical common-emitter current gain versus collector current curve. The low gain at low currents is due to the small recombination factor and the drop-off at the high current is due to the high-injection effect.

We now consider the second high-injection effect. At low injection, the majority carrier hole concentration at \( x = 0 \) for the npn transistor is

\[
p_p(0) = p_{p0} = N_a \tag{12.46a}
\]

and the minority carrier electron concentration is

\[
n_p(0) = n_{p0} \exp \left( \frac{eV_{BE}}{kT} \right) \tag{12.46b}
\]

The pn product is

\[
p_p(0)n_p(0) = p_{p0}n_{p0} \exp \left( \frac{eV_{BE}}{kT} \right) \tag{12.46c}
\]

At high injection, Equation (12.46c) still applies. However, \( p_p(0) \) will also increase, and for very high injection it will increase at nearly the same rate as \( n_p(0) \). The increase in \( n_p(0) \) will asymptotically approach the function

\[
n_p(0) \approx n_{p0} \exp \left( \frac{eV_{BE}}{2kT} \right) \tag{12.47}
\]