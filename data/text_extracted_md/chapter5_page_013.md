# CHAPTER 5 Carrier Transport Phenomena

Since mobility is a function of the ionized impurity concentration, we can use Figure 5.3 along with trial and error to determine \( \mu_n \) and \( N_d \). For example, if we choose \( N_d = 2 \times 10^{17} \), then \( N_i = N_d + N_a = 3 \times 10^{17} \) so that \( \mu_n \approx 510 \, \text{cm}^2/\text{V}\cdot\text{s} \) which gives \( \sigma = 8.16 \, (\Omega\cdot\text{cm})^{-1} \). If we choose \( N_d = 5 \times 10^{17} \), then \( N_i = 6 \times 10^{17} \) so that \( \mu_n \approx 325 \, \text{cm}^2/\text{V}\cdot\text{s} \), which gives \( \sigma = 20.8 \, (\Omega\cdot\text{cm})^{-1} \). The doping is bounded between these two values. Further trial and error yields

\[ N_d \approx 3.5 \times 10^{17} \, \text{cm}^{-3} \quad \text{and} \quad \mu_n \approx 400 \, \text{cm}^2/\text{V}\cdot\text{s} \]

which gives

\[ \sigma \approx 16 \, (\Omega\cdot\text{cm})^{-1} \]

**Comment**

We can see from this example that, in a high-conductivity semiconductor material, mobility is a strong function of carrier concentration.

## EXERCISE PROBLEM

**Ex 5.3** A compensated p-type silicon material at \( T = 300 \, \text{K} \) has impurity doping concentrations of \( N_i = 2.8 \times 10^{17} \, \text{cm}^{-3} \) and \( N_a = 8 \times 10^{16} \, \text{cm}^{-3} \). Determine the (a) hole mobility, (b) conductivity, and (c) resistivity.

----

## DESIGN EXAMPLE 5.4

**Objective:** Design a semiconductor resistor with a specified resistance to handle a given current density.

A silicon semiconductor at \( T = 300 \, \text{K} \) is initially doped with donors at a concentration of \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \). Acceptors are to be added to form a compensated p-type material. The resistor is to have a resistance of 10 k\(\Omega\) and handle a current density of 50 A/cm\(^2\) when 5 V is applied.

### Solution

For 5 V applied to a 10-k\(\Omega\) resistor, the total current is

\[ I = \frac{V}{R} = \frac{5}{10} = 0.5 \, \text{mA} \]

If the current density is limited to 50 A/cm\(^2\), then the cross-sectional area is

\[ A = \frac{I}{J} = \frac{0.5 \times 10^{-3}}{50} = 10^{-5} \, \text{cm}^2 \]

If we, somewhat arbitrarily at this point, limit the electric field to \( E = 100 \, \text{V/cm} \), then the length of the resistor is

\[ L = \frac{V}{E} = \frac{5}{100} = 5 \times 10^{-2} \, \text{cm} \]

From Equation (5.22b), the conductivity of the semiconductor is

\[ \sigma = \frac{L}{RA} = \frac{5 \times 10^{-2}}{(10)(10^{-5})} = 0.50 \, (\Omega\cdot\text{cm})^{-1} \]