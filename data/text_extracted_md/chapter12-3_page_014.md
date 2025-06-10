# Chapter 12: The Bipolar Transistor

!Simplified hybrid-pi equivalent circuit.

**Figure 12.4.1** Simplified hybrid-pi equivalent circuit.

## Solution

At very low frequency, we may neglect \( C_{\pi} \) so that

\[
V_{be} = I_b \, r_{\pi} \quad \text{and} \quad I_c = g_m V_{be} = g_m \, r_{\pi} \, I_b
\]

We can then write

\[
h_{fe0} = \frac{I_c}{I_b} = g_m r_{\pi}
\]

where \( h_{fe0} \) is the low-frequency, small-signal common-emitter current gain.

Taking into account \( C_{\pi} \), we have

\[
V_{be} = I_b \left( \frac{r_{\pi}}{1 + j \omega r_{\pi} C_{\pi}} \right)
\]

Then

\[
I_c = g_m V_{be} = I_b \left( \frac{h_{fe0}}{1 + j \omega r_{\pi} C_{\pi}} \right)
\]

or the small-signal current gain can be written as

\[
A_i = \frac{I_c}{I_b} = \frac{h_{fe0}}{1 + j \omega r_{\pi} C_{\pi}}
\]

The magnitude of the current gain is then

\[
|A_i| = \left| \frac{I_c}{I_b} \right| = \frac{h_{fe0}}{\sqrt{1 + (\omega r_{\pi} C_{\pi})^2}}
\]

The magnitude of the current gain drops to \( 1/\sqrt{2} \) of its low-frequency value at

\[
f = \frac{1}{2 \pi r_{\pi} C_{\pi}}
\]

If, for example, \( r_{\pi} = 2.6 \, \text{k}\Omega \) and \( C_{\pi} = 4 \, \text{pF} \), then

\[
f = 15.3 \, \text{MHz}
\]

## Comment

High-frequency transistors must have small-diffusion capacitances, implying the use of small devices.

## Exercise Problem

**Ex 12.13** Using the results of Example 12.13, determine the maximum value of \( C_{\pi} \) such that the frequency at which \( |A_i| = h_{fe0}/\sqrt{2} \) is \( f = 35 \, \text{MHz} \).