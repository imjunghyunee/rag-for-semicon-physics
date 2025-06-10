# CHAPTER 9 Metal–Semiconductor and Semiconductor Heterojunctions

!Figure 9.2

**Figure 9.2** Ideal energy-band diagram of a metal–semiconductor junction (a) under reverse bias and (b) under forward bias.

The energy-band diagrams versus voltage for the metal–semiconductor junction shown in Figure 9.2 are very similar to those of the pn junction given in the previous chapter. Because of this similarity, we expect the current–voltage characteristics of the Schottky barrier junction to be similar to the exponential behavior of the pn junction diode. The current mechanism here, however, is due to the flow of majority carrier electrons. In forward bias, the barrier seen by the electrons in the semiconductor is reduced, so majority carrier electrons flow more easily from the semiconductor into the metal. The forward-bias current is in the direction from metal to semiconductor: It is an exponential function of the forward-bias voltage \( V_a \).

## 9.1.2 Ideal Junction Properties

We can determine the electrostatic properties of the junction in the same way as we do for the pn junction. The electric field in the space charge region is determined from Poisson’s equation. We have that

\[
\frac{dE}{dx} = \frac{\rho(x)}{\varepsilon_s}
\]

(9.3)

where \( \rho(x) \) is the space charge volume density and \( \varepsilon_s \) is the permittivity of the semiconductor. If we assume that the semiconductor doping is uniform, then by integrating Equation (9.3), we obtain

\[
E = \int \frac{eN_d}{\varepsilon_s} dx = \frac{eN_d x}{\varepsilon_s} + C_1
\]

(9.4)

where \( C_1 \) is a constant of integration. The electric field is zero at the space charge edge in the semiconductor, so the constant of integration can be found as

\[
C_1 = -\frac{eN_d x_n}{\varepsilon_s}
\]

(9.5)