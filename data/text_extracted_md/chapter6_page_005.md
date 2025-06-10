## CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

Recombination rate for excess electrons is denoted by \( R'_n \) and for excess holes by \( R'_p \). Both parameters have units of \(\#/\text{cm}^3\cdot\text{s}\). The excess electrons and holes recombine in pairs, so the recombination rates must be equal. We can then write

\[
R'_n = R'_p
\tag{6.6}
\]

In the direct band-to-band recombination that we are considering, the recombination occurs spontaneously; thus, the probability of an electron and hole recombining is constant with time. The rate at which electrons recombine must be proportional to the electron concentration and must also be proportional to the hole concentration. If there are no electrons or holes, there can be no recombination.

The net rate of change in the electron concentration can be written as

\[
\frac{dn(t)}{dt} = \alpha_r \left[ n_i^2 - n(t)p(t) \right]
\tag{6.7}
\]

where

\[
n(t) = n_0 + \delta n(t)
\tag{6.8a}
\]

and

\[
p(t) = p_0 + \delta p(t)
\tag{6.8b}
\]

The first term, \(\alpha_r n_i^2\), in Equation (6.7) is the thermal-equilibrium generation rate. Since excess electrons and holes are created and recombine in pairs, we have that \(\delta n(t) = \delta p(t)\). (Excess electron and hole concentrations are equal so we can simply use the phrase excess carriers to mean either.) The thermal-equilibrium parameters, \(n_0\) and \(p_0\), are independent of time; therefore, Equation (6.7) becomes

\[
\frac{d[\delta n(t)]}{dt} = \alpha_r \left[ n_i^2 - (n_0 + \delta n(t))(p_0 + \delta p(t)) \right]
\]

\[
= -\alpha_r \delta n(t)(n_0 + p_0) + \delta n(t)
\tag{6.9}
\]

Equation (6.9) can easily be solved if we impose the condition of low-level injection. Low-level injection puts limits on the magnitude of the excess carrier concentration compared with the thermal-equilibrium carrier concentrations. In an extrinsic n-type material, we generally have \(n_0 \gg p_0\) and, in an extrinsic p-type material, we generally have \(p_0 \gg n_0\). Low-level injection means that the excess carrier concentration is much less than the thermal-equilibrium majority carrier concentration. Conversely, high-level injection occurs when the excess carrier concentration becomes comparable to or greater than the thermal-equilibrium majority carrier concentrations.

If we consider a p-type material (\(p_0 \gg n_0\)) under low-level injection (\(\delta n(t) \ll p_0\)), then Equation (6.9) becomes

\[
\frac{d[\delta n(t)]}{dt} = -\alpha_r p_0 \delta n(t)
\tag{6.10}
\]

The solution to the equation is an exponential decay from the initial excess concentration, or

\[
\delta n(t) = \delta n(0)e^{-\alpha_r p_0 t} = \delta n(0)e^{-t/\tau}
\tag{6.11}
\]