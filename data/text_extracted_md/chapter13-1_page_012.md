# CHAPTER 13 The Junction Field-Effect Transistor

!Figure 13.11 Simplified geometry of an n-channel pn JFET.

Pinchoff at the drain terminal occurs when \( h_2 = a \). At this point we reach what is known as the saturation condition; thus, we can write that \( V_{DS} = V_{DS}(sat) \). Then

\[
a = \left[ \frac{2 \varepsilon_s (V_{bi} + V_{DS}(sat) - V_{GS})}{eN_d} \right]^{1/2}
\]

(13.10)

This can be rewritten as

\[
V_{bi} + V_{DS}(sat) - V_{GS} = \frac{ea^2 N_d}{2 \varepsilon_s} = V_{p0}
\]

(13.11)

or

\[
V_{DS}(sat) = V_{p0} - (V_{bi} - V_{GS})
\]

(13.12)

Equation (13.12) gives the drain-to-source voltage to cause pinchoff at the drain terminal. The drain-to-source saturation voltage decreases with increasing reverse-biased gate-to-source voltage. We may note that Equation (13.12) has no meaning if \( |V_{GS}| > |V_t| \).

In a p-channel JFET, the voltage polarities are the reverse of those in the n-channel device. We can show that, in the p-channel JFET at saturation,

\[
V_{SD}(sat) = V_{p0} - (V_{bi} + V_{GS})
\]

(13.13)

where now the source is positive with respect to the drain.

## 13.2.2 Ideal DC Current–Voltage Relationship—Depletion Mode JFET

The derivation of the ideal current–voltage relation of the JFET is somewhat tedious, and the resulting equations are cumbersome in hand calculations. Before we go through this derivation, consider the following expression, which is a good approximation.