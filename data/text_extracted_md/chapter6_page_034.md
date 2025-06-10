# 6.5 Excess Carrier Lifetime

that \( R_n = R_p = 0 \). Equation (6.99), then, is the recombination rate of excess electrons and holes.

Since \( R \) in Equation (6.99) is the recombination rate of the excess carriers, we may write

\[
R = \frac{\delta n}{\tau}
\]

(6.100)

where \(\delta n\) is the excess carrier concentration and \(\tau\) is the lifetime of the excess carriers.

## 6.5.2 Limits of Extrinsic Doping and Low Injection

We simplified the ambipolar transport equation, Equation (6.39), from a nonlinear differential equation to a linear differential equation by applying limits of extrinsic doping and low injection. We may apply these same limits to the recombination rate equation.

Consider an n-type semiconductor under low injection. Then

\[
n_0 \gg p_0, \quad n_0 \gg \delta p, \quad n_0 \gg n', \quad n_0 \gg p'
\]

where \(\delta p\) is the excess minority carrier hole concentration. The assumptions of \(n_0 \gg n'\) and \(n_0 \gg p'\) imply that the trap level energy is near midgap so that \(n'\) and \(p'\) are not too different from the intrinsic carrier concentration. With these assumptions, Equation (6.99) reduces to

\[
R = C_n N_i \delta p
\]

(6.101)

The recombination rate of excess carriers in the n-type semiconductor is a function of the parameter \(C_n\), which is related to the minority carrier hole capture cross section. The recombination rate, then, is a function of the minority carrier parameter in the same way that the ambipolar transport parameters reduced to their minority carrier values.

The recombination rate is related to the mean carrier lifetime. Comparing Equations (6.100) and (6.101), we may write

\[
R = \frac{\delta n}{\tau} = C_n N_i \delta p = \frac{\delta p}{\tau_{p0}}
\]

(6.102)

where

\[
\tau_{p0} = \frac{1}{C_n N_i}
\]

(6.103)

and where \(\tau_{p0}\) is defined as the excess minority carrier hole lifetime. If the trap concentration increases, the probability of excess carrier recombination increases; thus, the excess minority carrier lifetime decreases.

Similarly, if we have a strongly extrinsic p-type material under low injection, we can assume that

\[
p_0 \gg n_0, \quad p_0 \gg \delta n, \quad p_0 \gg n', \quad p_0 \gg p'
\]