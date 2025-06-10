# Chapter 4: The Semiconductor in Equilibrium

The Fermi energy level for the intrinsic semiconductor is called the intrinsic Fermi energy, or \( E_F = E_{Fi} \). If we apply Equations (4.11) and (4.19) to the intrinsic semiconductor, then we can write

\[
n_0 = n_i = N_c \exp \left( \frac{-(E_c - E_{Fi})}{kT} \right)
\]
(4.20)

and

\[
p_0 = p_i = n_i = N_v \exp \left( \frac{-(E_{Fi} - E_v)}{kT} \right)
\]
(4.21)

If we take the product of Equations (4.20) and (4.21), we obtain

\[
n_i^2 = N_c N_v \exp \left( \frac{-(E_c - E_{Fi})}{kT} \right) \cdot \exp \left( \frac{-(E_{Fi} - E_v)}{kT} \right)
\]
(4.22)

or

\[
n_i^2 = N_c N_v \exp \left( \frac{-(E_c - E_v)}{kT} \right) = N_c N_v \exp \left( \frac{-E_g}{kT} \right)
\]
(4.23)

where \( E_g \) is the bandgap energy. For a given semiconductor material at a constant temperature, the value of \( n_i \) is a constant, and independent of the Fermi energy.

The intrinsic carrier concentration for silicon at \( T = 300 \, \text{K} \) may be calculated by using the effective density of states function values from Table 4.1. The value of \( n_i \) calculated from Equation (4.23) for \( E_g = 1.12 \, \text{eV} \) is \( n_i = 6.95 \times 10^9 \, \text{cm}^{-3} \). The commonly accepted value of \( n_i \) for silicon at \( T = 300 \, \text{K} \) is approximately \( 1.5 \times 10^{10} \, \text{cm}^{-3} \). This discrepancy may arise from several sources. First, the values of the effective masses are determined at a low temperature where the cyclotron resonance experiments are performed. Since the effective mass is an experimentally determined parameter, and since the effective mass is a measure of how well a particle moves in a crystal, this parameter may be a slight function of temperature. Next, the density of states function for a semiconductor was obtained by generalizing the model of an electron in a three-dimensional infinite potential well. This theoretical function may also not agree exactly with experiment. However, the difference between the theoretical value and the experimental value of \( n_i \) is approximately a factor of 2, which, in many cases, is not significant. Table 4.2 lists the commonly accepted values of \( n_i \) for silicon, gallium arsenide, and germanium at \( T = 300 \, \text{K} \).

The intrinsic carrier concentration is a very strong function of temperature.

### Table 4.2 | Commonly accepted values of \( n_i \) at \( T = 300 \, \text{K} \)

| Material        | \( n_i \) (\(\text{cm}^{-3}\)) |
|-----------------|-------------------------------|
| Silicon         | \( n_i = 1.5 \times 10^{10} \) |
| Gallium arsenide| \( n_i = 1.8 \times 10^6 \)    |
| Germanium       | \( n_i = 2.4 \times 10^{13} \) |

*Various references may list slightly different values of the intrinsic silicon concentration at room temperature. In general, they are all between \( 1 \times 10^{10} \) and \( 1.5 \times 10^{10} \, \text{cm}^{-3} \). This difference is, in most cases, not significant.*