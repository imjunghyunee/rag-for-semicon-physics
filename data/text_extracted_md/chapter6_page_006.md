where \(\tau_0 = ( \alpha_r p_0 )^{-1}\) and is a constant for the low-level injection. Equation (6.11) describes the decay of excess minority carrier electrons so that \(\tau_0\) is often referred to as the **excess minority carrier lifetime**. 

The recombination rate—which is defined as a positive quantity—of excess minority carrier electrons can be written, using Equation (6.10), as

\[
R'_n = -\frac{d(\delta n(t))}{dt} = \alpha_r p_0 \delta n(t) = \frac{\delta n(t)}{\tau_{n0}}
\]

(6.12)

For the direct band-to-band recombination, the excess majority carrier holes recombine at the same rate, so that for the p-type material

\[
R'_n = R'_p = \frac{\delta n(t)}{\tau_{n0}}
\]

(6.13)

In the case of an n-type material \((n_0 \gg p_0)\) under low-level injection \((\delta n(t) \ll n_0)\), the decay of minority carrier holes occurs with a time constant \(\tau_{p0} = (\alpha_r n_0)^{-1}\), where \(\tau_{p0}\) is also referred to as the excess minority carrier lifetime. The recombination rate of the majority carrier electrons will be the same as that of the minority carrier holes, so we have

\[
R'_n = R'_p = \frac{\delta n(t)}{\tau_{p0}}
\]

(6.14)

The generation rates of excess carriers are not functions of electron or hole concentrations. In general, the generation and recombination rates may be functions of the space coordinates and time.

----

### Objective: Determine the behavior of excess carriers as a function of time.

**EXAMPLE 6.1**

Assume that excess carriers have been generated uniformly in a semiconductor to a concentration of \(\delta n(0) = 10^{15} \, \text{cm}^{-3}\). The forcing function generating the excess carriers turns off at time \(t = 0\). Assuming the excess carrier lifetime is \(\tau_0 = 10^{-6} \, \text{s}\), determine \(\delta n(t)\) for \(t > 0\).

#### Solution

From Equation (6.11), we have

\[
\delta n(t) = \delta n(0) e^{-t/\tau_0} = 10^{15} e^{-t/10^{-6}} \, \text{cm}^{-3}
\]

For example, at:

- \(t = 0\), \(\delta n = 10^{15} \, \text{cm}^{-3}\)
- \(t = 1 \, \mu\text{s}\), \(\delta n = 10^{15} e^{-1} = 3.68 \times 10^{14} \, \text{cm}^{-3}\)
- \(t = 4 \, \mu\text{s}\), \(\delta n = 10^{15} e^{-4} = 1.83 \times 10^{13} \, \text{cm}^{-3}\)
- \(t = 10 \, \mu\text{s}\), \(\delta n = 10^{15} e^{-10} = 4.54 \times 10^{9} \, \text{cm}^{-3}\)

#### Comment

These results simply demonstrate the exponential decay of excess carriers with time after an excitation source is removed.

----

*In Chapter 5 we defined \(\tau\) as a mean time between collisions. We define \(\tau\) here as the mean time before a recombination event occurs. The two parameters are not related.*