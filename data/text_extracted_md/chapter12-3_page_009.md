## 12.5 Equivalent Circuit Models

Normally in electronic circuit applications, the collector–emitter voltage at saturation is of interest. We can define the C–E saturation voltage as

\[
V_{CE(sat)} = V_{BE} - V_{BC}
\]

(12.72)

We find an expression for \( V_{CE(sat)} \) by combining the Ebers–Moll equations. In the following example, we see how the Ebers–Moll equations can be used in a hand calculation, and we may also see how a computer analysis would make the calculations easier.

Combining Equations (12.64) and (12.70), we have

\[
-(I_B + I_C) = \alpha_F I_{CS} \left[ \exp \left( \frac{eV_{BE}}{kT} \right) - 1 \right] - I_{ES} \left[ \exp \left( \frac{eV_{BC}}{kT} \right) - 1 \right]
\]

(12.73)

If we solve for \([\exp(eV_{BE}/kT) - 1]\) from Equation (12.73), and substitute the resulting expression into Equation (12.68), we can then find \( V_{BE} \) as

\[
V_{BE} = V_T \ln \left[ \frac{I_C(1 - \alpha_R) + I_B + \frac{I_{CS}(1 - \alpha_F \alpha_R)}{I_{CS}(1 - \alpha_F \alpha_R)}} \right]
\]

(12.74)

where \( V_T \) is the thermal voltage. Similarly, if we solve for \([\exp(eV_{BE}/kT) - 1]\) from Equation (12.68), and substitute this expression into Equation (12.73), we can find \( V_{BC} \) as

\[
V_{BC} = V_T \ln \left[ \frac{\alpha_F I_B - (1 - \alpha_F) I_C + I_{CS}(1 - \alpha_R \alpha_F)}{I_{CS}(1 - \alpha_R \alpha_F)} \right]
\]

(12.75)

We may neglect the \( I_{ES} \) and \( I_{CS} \) terms in the numerators of Equations (12.74) and (12.75). Solving for \( V_{CE(sat)} \), we have

\[
V_{CE(sat)} = V_{BE} - V_{BC} = V_T \ln \left[ \frac{I_C(1 - \alpha_R) + I_B - \frac{I_{CS}}{I_{ES}}}{\frac{I_C}{\alpha_F} - (1 - \alpha_F) I_C - \frac{I_{CS}}{\alpha_R}} \right]
\]

(12.76)

The ratio of \( I_{CS} \) to \( I_{ES} \) can be written in terms of \( \alpha_F \) and \( \alpha_R \) from Equation (12.71). We can finally write

\[
V_{CE(sat)} = V_T \ln \left[ \frac{I_C(1 - \alpha_R) + I_B - \frac{I_{CS}}{\alpha_R}}{\alpha_F I_B - (1 - \alpha_F) I_C - \frac{I_{CS}}{\alpha_R}} \right]
\]

(12.77)

----

**Objective:** Calculate the collector–emitter saturation voltage of a bipolar transistor at \( T = 300 \, K \).

**Example 12.12**

Assume that \( \alpha_F = 0.99, \alpha_R = 0.20, I_C = 1 \, mA, \) and \( I_B = 50 \, \mu A \).

**Solution**

Substituting the parameters into Equation (12.77), we have

\[
V_{CE(sat)} = (0.0259) \ln \left[ \frac{(1)(1 - 0.2) + (0.05)}{(0.99)(0.05) - (1 - 0.99)(1)} \right] = 0.121 \, V
\]

**Comment**

This \( V_{CE(sat)} \) value is typical of collector–emitter saturation voltages. Because of the log function, \( V_{CE(sat)} \) is not a strong function of \( I_C \) or \( I_B \).