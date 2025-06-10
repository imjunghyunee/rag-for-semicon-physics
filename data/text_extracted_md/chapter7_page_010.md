```markdown
# CHAPTER 7 The pn Junction

The built-in potential barrier can be determined from Equation (7.10), and then the total space charge region width is obtained using Equation (7.31).

## EXAMPLE 7.2

**Objective:** Calculate the space charge width and electric field in a pn junction for zero bias.

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with doping concentrations of \( N_d = 10^{16} \, \text{cm}^{-3} \) and \( N_a = 10^{15} \, \text{cm}^{-3} \).

### Solution

In Example 7.1, we determined the built-in potential barrier as \( V_b = 0.635 \, \text{V} \). From Equation (7.31), the space charge width is

\[
W = \left[ \frac{2 \varepsilon_s V_b}{e} \left( \frac{N_a + N_d}{N_a N_d} \right) \right]^{1/2}
\]

\[
= \left[ \frac{2(11.7)(8.85 \times 10^{-14})(0.635)}{1.6 \times 10^{-19}} \left( \frac{10^{16} + 10^{15}}{(10^{16})(10^{15})} \right) \right]^{1/2}
\]

\[
= 0.951 \times 10^{-4} \, \text{cm} = 0.951 \, \mu\text{m}
\]

Using Equations (7.28) and (7.29), we can find \( x_n = 0.8644 \, \mu\text{m} \), and \( x_p = 0.0864 \, \mu\text{m} \).

The peak electric field at the metallurgical junction, using Equation (7.16) for example, is

\[
E_{\text{max}} = -\frac{eN_dx_n}{\varepsilon_s} = -\frac{(1.6 \times 10^{-19})(10^{16})(0.8644 \times 10^{-4})}{(11.7)(8.85 \times 10^{-14})} = -1.34 \times 10^{4} \, \text{V/cm}
\]

### Comment

The peak electric field in the space charge region of a pn junction is quite large. We must keep in mind, however, that there is no mobile charge in this region; hence there will be no drift current. We may also note, from this example, that the width of each space charge region is a reciprocal function of the doping concentration: The depletion region will extend further into the lower-doped region.

## EXERCISE PROBLEM

**Ex 7.2** A silicon pn junction at \( T = 300 \, \text{K} \) with zero applied bias has doping concentrations of \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{15} \, \text{cm}^{-3} \). Determine \( x_n, x_p, W, \) and \( E_{\text{max}} \).

## TEST YOUR UNDERSTANDING

**TYU 7.1** Calculate \( V_b, x_n, x_p, W, \) and \( E_{\text{max}} \) for a silicon pn junction at zero bias and \( T = 300 \, \text{K} \) for doping concentrations of (a) \( N_a = 2 \times 10^{17} \, \text{cm}^{-3}, N_d = 10^{16} \, \text{cm}^{-3} \) and (b) \( N_a = 4 \times 10^{15} \, \text{cm}^{-3}, N_d = 3 \times 10^{16} \, \text{cm}^{-3} \).

**TYU 7.2** Repeat Exercise Problem Ex 7.2 for a GaAs pn junction.
```