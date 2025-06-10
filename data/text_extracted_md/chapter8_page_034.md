# 8.3 Small-Signal Model of the pn Junction

where \( \delta p_n \) is the excess hole concentration in the n region. We are assuming that the ac signal voltage \( v_i (t) \) is sinusoidal. We then expect the steady-state solution for \( \delta p_n \) to be of the form of a sinusoidal solution superimposed on the dc solution, or

\[
\delta p_n(x, t) = \delta p_0(x) + p_1(x)e^{j\omega t}
\]

(8.76)

where \( \delta p_0(x) \) is the dc excess carrier concentration and \( p_1(x) \) is the magnitude of the ac component of the excess carrier concentration. The expression for \( \delta p_0(x) \) is the same as that given in Equation (8.14).

Substituting Equation (8.76) into the differential Equation (8.75), we obtain

\[
D_p \left[ \frac{\partial^2 [\delta p_0(x)]}{\partial x^2} + \frac{\partial^2 p_1(x)}{\partial x^2} e^{j\omega t} \right] - \frac{\delta p_0(x) + p_1(x)e^{j\omega t}}{\tau_0} = j\omega p_1(x)e^{j\omega t}
\]

(8.77)

We may rewrite this equation, combining the time-dependent and time-independent terms, as

\[
\left[ D_p \frac{\partial^2 [\delta p_0(x)]}{\partial x^2} - \frac{\delta p_0(x)}{\tau_0} \right] + \left[ D_p \frac{\partial^2 p_1(x)}{\partial x^2} - \frac{p_1(x)}{\tau_0} - j\omega p_1(x) \right] e^{j\omega t} = 0
\]

(8.78)

If the ac component, \( p_1(x) \), is zero, then the first bracketed term is just the differential Equation (8.10), which is identically zero. Then we have, from the second bracketed term,

\[
D_p \frac{d^2 p_1(x)}{dx^2} - \frac{p_1(x)}{\tau_0} - j\omega p_1(x) = 0
\]

(8.79)

Noting that \( L_p^2 = D_p \tau_0 \), Equation (8.79) may be rewritten in the form

\[
\frac{d^2 p_1(x)}{dx^2} - \frac{(1 + j\omega \tau_0)}{L_p^2} p_1(x) = 0
\]

(8.80)

or

\[
\frac{d^2 p_1(x)}{dx^2} - C_p^2 p_1(x) = 0
\]

(8.81)

where

\[
C_p^2 = \frac{(1 + j\omega \tau_0)}{L_p^2}
\]

(8.82)

The general solution to Equation (8.81) is

\[
p_1(x) = K_1 e^{-C_p x} + K_2 e^{C_p x}
\]

(8.83)

One boundary condition is that \( p_1(x \to + \infty) = 0 \), which implies that the coefficient \( K_2 = 0 \). Then

\[
p_1(x) = K_1 e^{-C_p x}
\]

(8.84)

Applying the boundary condition at \( x = 0 \) from Equation (8.74), we obtain

\[
p_1(0) = K_1 = p_{n0} \left( \frac{V_i}{V_t} \right)
\]

(8.85)