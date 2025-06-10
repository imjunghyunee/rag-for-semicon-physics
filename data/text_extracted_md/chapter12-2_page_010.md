# CHAPTER 12 The Bipolar Transistor

We then have

\[
0.9967 = \frac{1}{1 + \frac{10^{-8}}{10^{-11} \exp \left( -\frac{eV_{BE}}{2kT} \right)}}
\]

We can rearrange this equation and write

\[
\exp \left( \frac{+eV_{BE}}{2kT} \right) = \frac{0.9967 \times 10^3}{1 - 0.9967} = 3.02 \times 10^3
\]

Then

\[
V_{BE} = 2(0.0259) \ln (3.02 \times 10^3) = 0.654 \, \text{V}
\]

**Comment**

This example demonstrates that the recombination factor may be an important limiting factor in the bipolar current gain. In this example, if \( V_{BE} \) is smaller than 0.654 V, then the recombination factor \(\delta\) will fall below the desired 0.9967 value.

**EXERCISE PROBLEM**

Ex 12.6 If \( J_{B0} = 10^{-8} \, \text{A/cm}^2 \) and \( J_{E0} = 10^{-11} \, \text{A/cm}^2 \), determine the value of \( V_{BE} \) such that \(\delta = 0.9950\). (\( \lambda 0.0299 = 39A \, \mu V \))

----

## EXAMPLE 12.7

**Objective:** Calculate the common-emitter current gain of a silicon npn bipolar transistor at \( T = 300 \, \text{K} \) given a set of parameters.

Assume the following parameters:

- \( D_E = 10 \, \text{cm}^2/\text{s} \)
- \( D_B = 25 \, \text{cm}^2/\text{s} \)
- \( \tau_{nB} = 1 \times 10^{-7} \, \text{s} \)
- \( \tau_{pE} = 5 \times 10^{-7} \, \text{s} \)
- \( J_{E0} = 5 \times 10^{-14} \, \text{A/cm}^2 \)
- \( x_B = 0.70 \, \mu\text{m} \)
- \( x_E = 0.50 \, \mu\text{m} \)
- \( N_E = 1 \times 10^{18} \, \text{cm}^{-3} \)
- \( N_B = 1 \times 10^{16} \, \text{cm}^{-3} \)
- \( V_{BE} = 0.65 \, \text{V} \)

The following parameters are calculated:

\[
p_{B0} = \frac{(1.5 \times 10^{10})^2}{1 \times 10^{18}} = 2.25 \times 10^2 \, \text{cm}^{-3}
\]

\[
n_{E0} = \frac{(1.5 \times 10^{10})^2}{1 \times 10^{16}} = 2.25 \times 10^4 \, \text{cm}^{-3}
\]

\[
L_E = \sqrt{D_E \tau_{pE}} = 10^{-3} \, \text{cm}
\]

\[
L_B = \sqrt{D_B \tau_{nB}} = 3.54 \times 10^{-3} \, \text{cm}
\]

**Solution**

The emitter injection efficiency factor, from Equation (12.35a), is

\[
\gamma = \frac{1}{1 + \frac{(2.25 \times 10^4)(0.54 \times 10^{-3})}{(2.25 \times 10^2)(25 \times 10^{-5})} \cdot \frac{\tanh(0.0198)}{\tanh(0.050)}} = 0.9944
\]