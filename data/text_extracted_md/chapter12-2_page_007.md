# 12.3 Transistor Currents and Low-Frequency Common-Base Current Gain

Electrons toward the surface. This diffusion generates another component of recombination current and this component of recombination current must be included in the recombination factor δ. Although the actual calculation is difficult because of the two-dimensional analysis required, the form of the recombination current is the same as that of Equation (12.41).

## 12.3.3 Summary

Although we have considered an npn transistor in all of the derivations, exactly the same analysis applies to a pnp transistor; the same minority carrier distributions are obtained except that the electron concentrations become hole concentrations and vice versa. The current directions and voltage polarities also change.

We have been considering the common-base current gain, defined in Equation (12.27) as \( \alpha = I_C/I_E \). The common-emitter current gain is defined as \( \beta_0 = I_C/I_B \). From Figure 12.8 we see that \( I_E = I_B + I_C \). We can determine the relation between common-emitter and common-base current gains from the KCL equation. We can write

\[
\frac{I_E}{I_C} = \frac{I_B}{I_C} + 1
\]

Substituting the definitions of current gains, we have

\[
\frac{1}{\alpha_0} = \frac{1}{\beta_0} + 1
\]

Since this relation actually holds for both dc and small-signal conditions, we can drop the subscript. The common-emitter current gain can now be written in terms of the common-base current gain as

\[
\beta = \frac{\alpha}{1 - \alpha}
\]

The common-base current gain, in terms of the common-emitter current gain, is found to be

\[
\alpha = \frac{\beta}{1 + \beta}
\]

Table 12.3 summarizes the expressions for the limiting factors in the common-base current gain assuming that \( x_B \ll L_B \) and \( x_E^* \ll L_E \). Also given are the approximate expressions for the common-base current gain and the common-emitter current gain.

## 12.3.4 Example Calculations of the Gain Factors

If we assume a typical value of \( \beta \) to be 100, then \( \alpha = 0.99 \). If we also assume that \( \gamma = \alpha_T = \bar{\delta} \), then each factor would have to be equal to 0.9967 in order that \( \beta = 100 \). This calculation gives an indication of how close to unity each factor must be in order to achieve a reasonable current gain.