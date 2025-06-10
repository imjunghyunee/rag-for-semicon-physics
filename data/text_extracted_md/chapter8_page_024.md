# 8.2 Generation–Recombination Currents and High-Injection Levels

## Forward-Bias Recombination Current

For the reverse-biased pn junction, electrons and holes are essentially completely swept out of the space charge region so that \( n = p \approx 0 \). Under forward bias, however, electrons and holes are injected across the space charge region, so we do, in fact, have some excess carriers in the space charge region. The possibility exists that some of these electrons and holes will recombine within the space charge region and not become part of the minority carrier distribution.

The recombination rate of electrons and holes is again given from Equation (8.35) as

\[
R = \frac{C_n C_p N_i (np - n_i^2)}{C_n (n + n') + C_p (p + p')}
\]

Dividing both numerator and denominator by \( C_n C_p N_i \) and using the definitions of \( \tau_{n0} \) and \( \tau_{p0} \), we may write the recombination rate as

\[
R = \frac{np - n_i^2}{\tau_{p0} (n + n') + \tau_{n0} (p + p')} \tag{8.44}
\]

Figure 8.13 shows the energy-band diagram of the forward-biased pn junction. Shown in the figure are the intrinsic Fermi level and the quasi-Fermi levels for electrons and holes. From the results of Chapter 6, we may write the electron concentration as

\[
n = n_i \exp \left[ \frac{E_{Fn} - E_{Fi}}{kT} \right] \tag{8.45}
\]

and the hole concentration as

\[
p = n_i \exp \left[ \frac{E_{Fi} - E_{Fp}}{kT} \right] \tag{8.46}
\]

where \( E_{Fn} \) and \( E_{Fp} \) are the quasi-Fermi levels for electrons and holes, respectively.

!Energy-band diagram

**Figure 8.13** | Energy-band diagram of a forward-biased pn junction including quasi-Fermi levels.