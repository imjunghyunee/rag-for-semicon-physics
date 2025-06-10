## 14.3 Photodetectors

### Objective

Calculate the photoconductor gain of a silicon photoconductor.

Consider an n-type silicon photoconductor with a length \( L = 100 \, \mu m \), cross-sectional area \( A = 10^{-7} \, cm^2 \), and minority carrier lifetime \( \tau_r = 10^{-6} \, s \). Let the applied voltage be \( V = 10 \) volts.

### Solution

The electron transit time is determined as

\[
t_e = \frac{L}{\mu_e E} = \frac{L^2}{\mu_e V} = \frac{(100 \times 10^{-4})^2}{(1350)(10)} = 7.41 \times 10^{-9} \, s
\]

The photoconductor gain is then

\[
\Gamma_{ph} = \frac{\tau_r}{t_e} \left( 1 + \frac{\mu_h}{\mu_e} \right) = \frac{10^{-6}}{7.41 \times 10^{-9}} \left( 1 + \frac{480}{1350} \right) = 1.83 \times 10^2
\]

### Comment

The fact that a photoconductor—a bar of semiconductor material—has a gain may be surprising.

### EXERCISE PROBLEM

**Ex 14.4** Consider the photoconductor described in Example 14.4. Determine the photocurrent if \( G_t = 10^{21} \, cm^{-3} \cdot s^{-1} \) and \( E = 10 \, V/cm \). Also assume that \( \mu_e = 1000 \, cm^2/V \cdot s \) and \( \mu_p = 400 \, cm^2/V \cdot s \).

\((\text{Verify } \Gamma_{ph} = 7.1 \times 10^2)\)

----

Let's consider physically what happens to a photon-generated electron, for example. After the excess electron is generated, it drifts very quickly out of the photoconductor at the anode terminal. In order to maintain charge neutrality throughout the photoconductor, another electron immediately enters the photoconductor at the cathode and drifts toward the anode. This process will continue during a time period equal to the mean carrier lifetime. At the end of this period, on the average, the photoelectron will recombine with a hole.

The electron transit time, using the parameters from Example 14.4, is \( t_e = 7.41 \times 10^{-9} \, s \). In a simplistic sense, the photoelectron will circulate around the photoconductor circuit 135 times during the \( 10^{-6} \, s \) time duration, which is the mean carrier lifetime. If we take into account the photon-generated hole, the total number of charges collected at the photoconductor contacts for every electron generated is 183.

When the optical signal ends, the photocurrent will decay exponentially with a time constant equal to the minority carrier lifetime. From the photoconductor gain expression, we would like a large minority carrier lifetime, but the switching speed is enhanced by a small minority carrier lifetime. There is obviously a trade-off between gain and speed. In general, the performance of a photodiode, which we will discuss next, is superior to that of a photoconductor.

### 14.3.2 Photodiode

A photodiode is a pn junction diode operated with an applied reverse-biased voltage. We will initially consider a long diode in which excess carriers are generated.