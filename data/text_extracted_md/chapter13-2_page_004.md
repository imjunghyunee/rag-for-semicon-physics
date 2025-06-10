# 13.3.1 Channel Length Modulation

The expression for the drain current is inversely proportional to the channel length \( L \) as given, for example, by Equation (13.27). In deriving the current equations, we have implicitly assumed that the channel length was constant. However, the effective channel length can change. Figure 13.5 shows the space charge region in the channel when the transistor is biased in the saturation region. The neutral n-channel length decreases as \( V_{DS} \) increases; thus, the drain current will increase. The change in the effective channel length and the corresponding change in drain current is called channel length modulation.

The pinchoff current, Equation (13.28), is modified by the channel length modulation and can be written as

\[
I_{p1} = \frac{\mu_n (eN_d)^2 W a^3}{6 \epsilon_s L'}
\]

(13.50)

where

\[
L' \approx L - \frac{1}{2} \Delta L
\]

(13.51)

If we assume the channel depletion region shown in Figure 13.5 extends equally into the channel and drain regions, then as a first approximation, we will include the factor \(\frac{1}{2}\) in the expression for \( L' \).

The drain current can be written as

\[
I_{D1} = I_{D0} \frac{I_{p1}}{I_{p1}} = I_{D0} \left( \frac{L}{L - \frac{1}{2} \Delta L} \right)
\]

(13.52)

where \( I_{D0} \) is the ideal drain current predicted by Equation (13.35). Another form of the current–voltage characteristic in the saturation region is given by

\[
I_{D1} (sat) = I_{D0} (sat)(1 + \lambda V_{DS})
\]

(13.53)

The effective channel length \( L' \) supports the \( V_{DS}(sat) \) voltage, and the space charge region length \( \Delta L \) in the channel supports the drain voltage beyond the saturation value. Neglecting charges in the space charge region due to current flow, the depletion length \( \Delta L \) is then, to a first approximation, given by

\[
\Delta L = \left[ \frac{2 \epsilon_s (V_{DS} - V_{DS}(sat))}{eN_d} \right]^{1/2}
\]

(13.54)