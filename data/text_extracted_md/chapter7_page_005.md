# 7.2 Zero Applied Bias

Combining Equations (7.7) and (7.8), we find that

\[
\phi_{fp} = \frac{kT}{e} \ln \left( \frac{N_a}{n_i} \right)
\]

(7.9)

Finally, the built-in potential barrier for the step junction is found by substituting Equations (7.6) and (7.9) into Equation (7.1), which yields

\[
V_0 = \frac{kT}{e} \ln \left( \frac{N_a N_d}{n_i^2} \right) = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right)
\]

(7.10)

where \( V_t = kT/e \) and is defined as the thermal voltage.

At this time, we should note a subtle but important point concerning notation. Previously in the discussion of a semiconductor material, \( N_d \) and \( N_a \) denoted donor and acceptor impurity concentrations in the same region, thereby forming a compensated semiconductor. From this point on in the text, \( N_d \) and \( N_a \) will denote the net donor and acceptor concentrations in the individual n and p regions, respectively. If the p region, for example, is a compensated material, then \( N_a \) will represent the difference between the actual acceptor and donor impurity concentrations. The parameter \( N_d \) is defined in a similar manner for the n region.

----

**Objective:** Calculate the built-in potential barrier in a pn junction.

Consider a silicon pn junction at \( T = 300 \, K \) with doping concentrations of \( N_d = 2 \times 10^{17} \, \text{cm}^{-3} \) and \( N_a = 10^{15} \, \text{cm}^{-3} \).

**Solution**

The built-in potential barrier is determined from Equation (7.10) as

\[
V_0 = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left( \frac{2 \times 10^{17} \times 10^{15}}{(1.5 \times 10^{10})^2} \right) = 0.713 \, V
\]

If we change the doping concentration in the p region of the pn junction such that the doping concentrations become \( N_a = 10^{16} \, \text{cm}^{-3} \) and \( N_d = 10^{15} \, \text{cm}^{-3} \), then the built-in potential barrier becomes \( V_0 = 0.635 \, V \).

**Comment**

The built-in potential barrier changes only slightly as the doping concentrations change by orders of magnitude because of the logarithmic dependence.

**EXERCISE PROBLEM**

Ex 7.1 (a) Calculate the built-in potential barrier in a silicon pn junction at \( T = 300 \, K \) for 
(i) \( N_a = 5 \times 10^{15} \, \text{cm}^{-3} \), \( N_d = 10^{17} \, \text{cm}^{-3} \) and (ii) \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \), \( N_d = 2 \times 10^{15} \, \text{cm}^{-3} \).

(b) Repeat part (a) for a GaAs pn junction.