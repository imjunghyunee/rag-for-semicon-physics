```markdown
# 4.1 Charge Carriers in Semiconductors

## Objective: Calculate the intrinsic carrier concentration in silicon at \( T = 250 \, \text{K} \) and at \( T = 400 \, \text{K} \)

The values of \( N_c \) and \( N_v \) for silicon at \( T = 300 \, \text{K} \) are \( 2.8 \times 10^{19} \, \text{cm}^{-3} \) and \( 1.04 \times 10^{19} \, \text{cm}^{-3} \), respectively. Both \( N_c \) and \( N_v \) vary as \( T^{3/2} \). Assume the bandgap energy of silicon is 1.12 eV and does not vary over this temperature range.

### Solution

Using Equation (4.23), we find, at \( T = 250 \, \text{K} \)

\[
n_i^2 = (2.8 \times 10^{19} \times 1.04 \times 10^{19}) \left(\frac{250}{300}\right)^3 \exp\left(\frac{-1.12}{(0.0259)(250/300)}\right)
\]

\[
= 4.90 \times 10^{15}
\]

or

\[
n_i = 7.0 \times 10^7 \, \text{cm}^{-3}
\]

At \( T = 400 \, \text{K} \), we find

\[
n_i^2 = (2.8 \times 10^{19} \times 1.04 \times 10^{19}) \left(\frac{400}{300}\right)^3 \exp\left(\frac{-1.12}{(0.0259)(400/300)}\right)
\]

\[
= 5.67 \times 10^{24}
\]

or

\[
n_i = 2.38 \times 10^{12} \, \text{cm}^{-3}
\]

### Comment

We may note from this example that the intrinsic carrier concentration increased by over 4 orders of magnitude as the temperature increased by 150°C.

### EXERCISE PROBLEM

**Ex 4.3** (a) Calculate the intrinsic carrier concentration in GaAs at \( T = 400 \, \text{K} \) and at \( T = 250 \, \text{K} \). Assume that \( E_g = 1.42 \, \text{eV} \) is constant over this temperature range.  
(b) What is the ratio of \( n_i \) at \( T = 400 \, \text{K} \) to that at \( T = 250 \, \text{K} \)?

\[
[01 \times 10^{19} \, \text{cm}^{-3} \times 1.01 \times 10^{19} \, \text{cm}^{-3} \times (0.0259)(1/u) \times 1.01 \times 6.2 \times (00)^{1/2} \, \text{cm}^{-3}]
\]

Figure 4.2 is a plot of \( n_i \) from Equation (4.23) for silicon, gallium arsenide, and germanium as a function of temperature. As seen in the figure, the value of \( n_i \) for these semiconductors may easily vary over several orders of magnitude as the temperature changes over a reasonable range.

----

### TEST YOUR UNDERSTANDING

**TYU 4.3** Calculate the intrinsic concentration in silicon at (a) \( T = 200 \, \text{K} \) and (b) \( T = 450 \, \text{K} \). Determine the ratio of \( n_i \) at \( T = 450 \, \text{K} \) to that at \( T = 200 \, \text{K} \).

\[
[01 \times 9.2 \times (1) \times 1.01 \times 1 \times 1.01 \times 9.9 \times 1 = 1 \, \text{cm}^{-3}]
\]

**TYU 4.4** Repeat TYU 4.3 for GaAs.

\[
[01 \times 1.82 \times (1) \times 1.01 \times 9.8 \times 1 = 1 \, \text{cm}^{-3}]
\]

**TYU 4.5** Repeat TYU 4.3 for Ge.

\[
[01 \times 8.1 \times (1) \times 1.01 \times 6.2 \times 1 = 1 \, \text{cm}^{-3}]
\]
```