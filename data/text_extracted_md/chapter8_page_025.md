```markdown
# CHAPTER 8 The pn Junction Diode

!Figure 8.14

**Figure 8.14** Relative magnitude of the recombination rate through the space charge region of a forward-biased pn junction.

From Figure 8.13, we may note that

\[
(E_{Fn} - E_{Fi}) + (E_{Fp} - E_{Fi}) = eV_a
\]

(8.47)

where \( V_a \) is the applied forward-bias voltage. Again, if we assume that the trap level is at the intrinsic Fermi level, then \( n' = p' = n_i \). Figure 8.14 shows a plot of the relative magnitude of the recombination rate as a function of distance through the space charge region. This plot was generated using Equations (8.44), (8.45), (8.46), and (8.47). A very sharp peak occurs at the metallurgical junction (\( x = 0 \)).

At the center of the space charge region, we have

\[
E_{Fn} - E_{Fi} = E_{Fi} - E_{Fp} = \frac{eV_a}{2}
\]

(8.48)

Equations (8.45) and (8.46) then become

\[
n = n_i \exp \left( \frac{eV_a}{2kT} \right)
\]

(8.49)

and

\[
p = n_i \exp \left( \frac{eV_a}{2kT} \right)
\]

(8.50)

If we assume that \( n' = p' = n_i \) and that \( \tau_{n0} = \tau_{p0} = \tau_0 \), then Equation (8.44) becomes

\[
R_{max} = \frac{n_i}{2\tau_0} \left[ \frac{\exp(eV_a/kT) - 1}{\exp(eV_a/2kT) + 1} \right]
\]

(8.51)

which is the maximum recombination rate for electrons and holes that occurs at the center of the forward-biased pn junction. If we assume that \( V_a \gg kT/e \), we may
```