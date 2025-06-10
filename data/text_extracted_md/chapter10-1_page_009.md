# 10.1 The Two-Terminal MOS Structure

The maximum space charge width, \( x_{\text{scr}} \), at this inversion transition point can be calculated from Equation (10.5) by setting \(\Phi_s = 2\Phi_p\). Then

\[
x_{\text{scr}} = \left( \frac{4 \varepsilon_s \Phi_p}{e N_a} \right)^{1/2}
\]

(10.6)

----

## Objective

Calculate the maximum space charge width for a given semiconductor doping concentration.

Consider silicon at \( T = 300 \, \text{K} \) doped to \( N_a = 10^{16} \, \text{cm}^{-3} \). The intrinsic carrier concentration is \( n_i = 1.5 \times 10^{10} \, \text{cm}^{-3} \).

### Solution

From Equation (10.4), we have

\[
\Phi_p = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{10^{16}}{1.5 \times 10^{10}} \right) = 0.3473 \, \text{V}
\]

Then the maximum space charge width is

\[
x_{\text{scr}} = \left( \frac{4 \varepsilon_s \Phi_p}{e N_a} \right)^{1/2} = \left[ \frac{4(11.7)(8.85 \times 10^{-14})(0.3473)}{(1.6 \times 10^{-19})(10^{16})} \right]^{1/2}
\]

or

\[
x_{\text{scr}} \approx 0.30 \times 10^{-4} \, \text{cm} = 0.30 \, \mu\text{m}
\]

### Comment

The maximum induced space charge width is on the same order of magnitude as pn junction space charge widths.

### Exercise Problem

**Ex 10.1** Consider an oxide-to-p-type silicon junction at \( T = 300 \, \text{K} \). The impurity doping concentration in the silicon is \( N_s = 2 \times 10^{15} \, \text{cm}^{-3} \). Calculate the maximum space charge width. Does the space charge width increase or decrease as the p-type doping concentration decreases?

----

We have been considering a p-type semiconductor substrate. The same maximum induced space charge region width occurs in an n-type substrate. Figure 10.10 is the energy-band diagram at the threshold voltage with an n-type substrate. We can write

\[
\Phi_n = V_t \ln \left( \frac{N_d}{n_i} \right)
\]

(10.7)

and

\[
x_{\text{scr}} = \left( \frac{4 \varepsilon_s \Phi_n}{e N_d} \right)^{1/2}
\]

(10.8)

Note that we are always assuming the parameters \(\Phi_s\) and \(\Phi_n\) to be positive quantities.

Figure 10.11 is a plot of \( x_{\text{scr}} \) at \( T = 300 \, \text{K} \) as a function of doping concentration in silicon. The semiconductor doping can be either n type or p type.