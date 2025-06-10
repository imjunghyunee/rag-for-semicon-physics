# Chapter 12: The Bipolar Transistor

The transit time can then be found by integrating, or

\[
\tau_b = \int_0^{x_b} \frac{dx}{v(x)} = \int_0^{x_b} \frac{n_B(x) \, dx}{(-J_n)}
\]

(12.91)

The electron concentration in the base is approximately linear (see Equation (12.15b)) so we can write

\[
n_B(x) \approx n_{B0} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) \right] \left( 1 - \frac{x}{x_B} \right)
\]

(12.92)

and the electron current density is given by

\[
J_n = eD_n \frac{dn_B(x)}{dx}
\]

(12.93)

The base transit time is then found by combining Equations (12.92) and (12.93) with Equation (12.91). We find that

\[
\tau_b = \frac{x_B^2}{2D_n}
\]

(12.94)

The third time-delay factor is \(\tau_d\), the collector depletion region transit time. Assuming that the electrons in the npn device travel across the B–C space charge region at their saturation velocity, we have

\[
\tau_d = \frac{x_{dc}}{v_s}
\]

(12.95)

where \(x_{dc}\) is the B–C space charge width and \(v_s\) is the electron saturation velocity.

The fourth time-delay factor, \(\tau_c\), is the collector capacitance charging time. The B–C is reverse biased so that the diffusion resistance in parallel with the junction capacitance is very large. The charging time constant is then a function of the collector series resistance \(r_c\). We can write

\[
\tau_c = r_c(C_{jc} + C_s)
\]

(12.96)

where \(C_{jc}\) is the B–C junction capacitance and \(C_s\) is the collector-to-substrate capacitance. The series resistance in small epitaxial transistors is usually small; thus, the time delay \(\tau_c\) may be neglected in some cases.

Example calculations of the various time-delay factors are given in the next section as part of the cutoff frequency discussion.

## 12.6.2 Transistor Cutoff Frequency

The current gain as a function of frequency is developed in Example 12.13 so that we can also write the common-base current gain as

\[
\alpha = \frac{\alpha_0}{1 + j\frac{f}{f_{\alpha}}}
\]

(12.97)