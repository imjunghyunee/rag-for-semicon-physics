# The Junction Field-Effect Transistor

Then Equation (13.20b) becomes

\[
I_{D1} \, dx = \frac{\mu_n (eN_d)^2 W}{\epsilon_s} \left[ ah(x) \, dh(x) - h(x)^2 \, dh(x) \right]
\]

(13.23)

The drain current \( I_{D1} \) is found by integrating Equation (13.23) along the channel length. Assuming the current and mobility are constant through the channel, we obtain

\[
I_{D1} = \frac{\mu_n (eN_d)^2 W}{\epsilon_s L} \left[ \int_{h_m}^{h_2} ah \, dh - \int_{h_m}^{h_2} h^2 \, dh \right]
\]

(13.24)

or

\[
I_{D1} = \frac{\mu_n (eN_d)^2 W}{\epsilon_s L} \left[ \frac{a}{2} (h_2^2 - h_m^2) - \frac{1}{3} (h_2^3 - h_m^3) \right]
\]

(13.25)

Noting that

\[
h_2^2 = \frac{2 \epsilon_s (V_{DS} + V_{bi} - V_{GS})}{eN_d}
\]

(13.26a)

\[
h_m^2 = \frac{2 \epsilon_s (V_{bi} - V_{GS})}{eN_d}
\]

(13.26b)

and

\[
V_{po} = \frac{ea^2 N_d}{2 \epsilon_s}
\]

(13.26c)

Equation (13.25) can be written as

\[
I_{D1} = \frac{\mu_n (eN_d)^2 Wa^3}{2 \epsilon_s L} \left[ \frac{V_{DS}}{V_{po}} - \frac{2}{3} \left( \frac{V_{DS} + V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} + \frac{2}{3} \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} \right]
\]

(13.27)

We may define

\[
I_{p1} = \frac{\mu_n (eN_d)^2 Wa^3}{6 \epsilon_s L}
\]

(13.28)

where \( I_{p1} \) is called the pinch-off current. Equation (13.27) becomes

\[
I_{D1} = I_{p1} \left[ 3 \left( \frac{V_{DS}}{V_{po}} \right) - 2 \left( \frac{V_{DS} + V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} + 2 \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right)^{3/2} \right]
\]

(13.29)

Equation (13.29) is valid for \( 0 \leq |V_{GS}| \leq |V_t| \) and \( 0 \leq V_{DS} \leq V_{DS(sat)} \). The pinch-off current \( I_{p1} \) would be the maximum drain current in the JFET if the zero-biased depletion regions could be ignored or if \( V_{GS} \) and \( V_t \) were both zero.

Equation (13.29) is the current–voltage relationship for the one-sided n-channel JFET in the nonsaturation region. For the two-sided symmetrical JFET shown in Figure 13.9a, the total drain current would be \( I_{D2} = 2I_{D1} \).