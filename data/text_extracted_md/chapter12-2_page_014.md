# CHAPTER 12 The Bipolar Transistor

and

\[
|J_c| = \frac{(1.6 \times 10^{-19})(25)(4.5 \times 10^{9})}{0.6482 \times 10^{-4}} \exp\left(\frac{0.60}{0.0259}\right) = 3.195 \, \text{A/cm}^2
\]

For \( V_{CE} = 10 \, \text{V} \), we find (see the following Exercise Problem Ex 12.8)

\[
x_B = 0.70 - 0.103 = 0.597 \, \mu \text{m}
\]

and

\[
|J_c| = \frac{(1.6 \times 10^{-19})(25)(4.5 \times 10^{9})}{0.597 \times 10^{-4}} \exp\left(\frac{0.60}{0.0259}\right) = 3.469 \, \text{A/cm}^2
\]

We now can find, from Equation (12.45a)

\[
\frac{dI_C}{dV_{CE}} = \frac{\Delta I_C}{\Delta V_{CE}} = \frac{J_C}{V_{CE} + V_A} = \frac{J_C}{V_{BE} + V_{CE} + V_A}
\]

or

\[
\frac{3.469 - 3.195}{8} = \frac{3.195}{0.60 + 2 + V_A}
\]

The Early voltage is then determined to be

\[
V_A = 90.7 \, \text{V}
\]

## Comment

This example indicates how much the collector current can change as the neutral base width changes with a change in the B–C space charge width, and it also indicates the magnitude of the Early voltage.

## EXERCISE PROBLEM

**Ex 12.8** Consider a silicon npn bipolar transistor with parameters described in Example 12.8. Determine the neutral base width for a C–B voltage of  
(a) \( V_{CB} = 2 \, \text{V} \) and (b) \( V_{CB} = 10 \, \text{V} \). Neglect the B–E space charge width.

The previous example and exercise problem demonstrate, too, that we can expect variations in transistor properties due to tolerances in transistor-fabrication processes. There will be variations, in particular, in the base width of narrow-base transistors that will cause variations in the collector current characteristics simply due to the tolerances in processing.

## 12.4.2 High Injection

The ambipolar transport equation that we have used to determine the minority carrier distributions assumed low injection. As \( V_{BE} \) increases, the injected minority carrier concentration may approach, or even become larger than, the majority carrier concentration. If we assume quasi–charge neutrality, then the majority carrier hole concentration in the p-type base at \( x = 0 \) will increase as shown in Figure 12.23 because of the excess holes.