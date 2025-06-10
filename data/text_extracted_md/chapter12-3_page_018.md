# CHAPTER 12 The Bipolar Transistor

!Bode plot

**Figure 12.42** Bode plot of common-emitter current gain versus frequency.

Comparing Equations (12.104) and (12.102), the beta cutoff frequency is related to the cutoff frequency by

\[
f_{\beta} = \frac{f_T}{\beta_0}
\]

(12.105)

Figure 12.42 shows a Bode plot of the common-emitter current gain as a function of frequency and shows the relative values of the beta and cutoff frequencies. Keep in mind that the frequency is plotted on a log scale, so \( f_{\beta} \) and \( f_T \) usually have significantly different values.

----

## EXAMPLE 12.14

**Objective:** Calculate the emitter-to-collector transit time and the cutoff frequency of a bipolar transistor, with the following parameters.

Consider a silicon npn transistor at \( T = 300 \, K \). Assume the following parameters:

- \( I_E = 1 \, \text{mA} \)
- \( x_B = 0.5 \, \mu \text{m} \)
- \( x_{de} = 2.4 \, \mu \text{m} \)
- \( C_{\mu} = 0.1 \, \text{pF} \)
- \( C_{\pi} = 1 \, \text{pF} \)
- \( D_n = 25 \, \text{cm}^2/\text{s} \)
- \( r_e = 20 \, \Omega \)
- \( C_c = 0.1 \, \text{pF} \)

### Solution

We will initially calculate the various time-delay factors. If we neglect the parasitic capacitance, the emitter–base junction charging time is

\[
\tau_{\pi} = r'_e C_{\pi}
\]

where

\[
r'_e = \frac{kT}{e} \cdot \frac{1}{I_E} = \frac{0.0259}{1 \times 10^{-3}} = 25.9 \, \Omega
\]

Then

\[
\tau_{\pi} = (25.9)(10^{-12}) = 25.9 \, \text{ps}
\]

The base transit time is

\[
\tau_B = \frac{x_B^2}{2D_n} = \frac{(0.5 \times 10^{-4})^2}{2(25)} = 50 \, \text{ps}
\]