# 5.1 Carrier Drift

The conductivity of a compensated p-type semiconductor is

\[
\sigma \approx e\mu_p p = e\mu_p (N_a - N_d)
\]

where the mobility is a function of the total ionized impurity concentration \(N_a + N_d\).

Using trial and error, if \(N_a = 1.25 \times 10^{16} \, \text{cm}^{-3}\), then \(N_a + N_d = 1.75 \times 10^{16} \, \text{cm}^{-3}\), and the hole mobility, from Figure 5.3, is approximately \(\mu_p = 410 \, \text{cm}^2/\text{V}\cdot\text{s}\). The conductivity is then

\[
\sigma = e\mu_p(N_a - N_d) = (1.6 \times 10^{-19})(410)(1.25 \times 10^{16} - 5 \times 10^{15}) = 0.492
\]

which is very close to the value we need.

## Comment

Since the mobility is related to the total ionized impurity concentration, the determination of the impurity concentration to achieve a particular conductivity is not straightforward.

## EXERCISE PROBLEM

**Ex 5.4** A bar of p-type silicon, such as shown in Figure 5.5, has a cross-sectional area \(A = 10^{-6} \, \text{cm}^2\) and a length \(L = 1.2 \times 10^{-1} \, \text{cm}\). For an applied voltage of 5 V, a current of 2 mA is required. What is the required (a) resistance, (b) resistivity, and (c) impurity doping concentration? (d) What is the resulting hole mobility?

----

For an intrinsic material, the conductivity can be written as

\[
\sigma_i = e(\mu_n + \mu_p) n_i \tag{5.25}
\]

The concentrations of electrons and holes are equal in an intrinsic semiconductor, so the intrinsic conductivity includes both the electron and hole mobility. Since, in general, the electron and hole mobilities are not equal, the intrinsic conductivity is not the minimum value possible at a given temperature.

## 5.1.4 Velocity Saturation

So far in our discussion of drift velocity, we have assumed that mobility is not a function of electric field, meaning that the drift velocity will increase linearly with applied electric field. The total velocity of a particle is the sum of the random thermal velocity and drift velocity. At \(T = 300 \, \text{K}\), the average random thermal energy is given by

\[
\frac{1}{2} m_n v_t^2 = \frac{3}{2} kT = \frac{3}{2} (0.0259) = 0.03885 \, \text{eV} \tag{5.26}
\]

This energy translates into a mean thermal velocity of approximately \(10^7 \, \text{cm/s}\) for an electron in silicon. If we assume an electron mobility of \(\mu_n = 1350 \, \text{cm}^2/\text{V}\cdot\text{s}\) in low-doped silicon, a drift velocity of \(10^5 \, \text{cm/s}\), or 1 percent of the thermal velocity, is achieved if the applied electric field is approximately \(75 \, \text{V/cm}\). This applied electric field does not appreciably alter the energy of the electron.

Figure 5.7 is a plot of average drift velocity as a function of applied electric field for electrons and holes in silicon, gallium arsenide, and germanium. At low electric fields, where there is a linear variation of velocity with electric field, the slope of