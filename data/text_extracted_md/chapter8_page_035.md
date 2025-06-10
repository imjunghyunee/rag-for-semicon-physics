# CHAPTER 8 The pn Junction Diode

The hole diffusion current density can be calculated at \( x = 0 \). This is given by

\[
J_p = -eD_p \frac{\partial p}{\partial x} \bigg|_{x=0}
\]

(8.86)

If we consider a homogeneous semiconductor, the derivative of the hole concentration will be just the derivative of the excess hole concentration. Then

\[
J_p = -eD_p \frac{\partial (\delta p)}{\partial x} \bigg|_{x=0} = -eD_p \frac{\partial [\delta p(x)]}{\partial x} \bigg|_{x=0} = -eD_p \frac{\partial p(x)}{\partial x} \bigg|_{x=0} e^{j\omega t}
\]

(8.87)

We can write this equation in the form

\[
J_p = J_{p_0} + j_p(t)
\]

(8.88)

where

\[
J_{p_0} = -eD_p \frac{\partial [\delta p(x)]}{\partial x} \bigg|_{x=0} = \frac{eD_p p_{n_0}}{L_p} \left[ \exp \left( \frac{eV_0}{kT} \right) - 1 \right]
\]

(8.89)

Equation (8.89) is the dc component of the hole diffusion current density and is exactly the same as in the ideal I–V relation derived previously.

The sinusoidal component of the diffusion current density is then found from

\[
j_p(t) = \hat{j}_p e^{j\omega t} = -eD_p \frac{\partial p(x)}{\partial x} \bigg|_{x=0} e^{j\omega t}
\]

(8.90)

where \(\hat{j}_p\) is the current density phasor. Combining Equations (8.90), (8.84), and (8.85), we have

\[
\hat{J}_p = -eD_p (A - C_p) \left[ \frac{\hat{V}_i}{V_T} \right] e^{-j\omega t} \bigg|_{x=0}
\]

(8.91)

We can write the total ac hole current phasor as

\[
\hat{I}_p = A \hat{j}_p = eAD_p C_p \frac{\hat{v}_i}{V_T}
\]

(8.92)

where \( A \) is the cross-sectional area of the pn junction. Substituting the expression for \( C_p \), we obtain

\[
\hat{I}_p = \frac{eAD_p p_{n_0}}{L_p} \sqrt{1 + j\omega \tau_0} \left( \frac{\hat{v}_i}{V_T} \right)
\]

(8.93)

If we define

\[
I_{p_0} = \frac{eAD_p p_{L}}{L_p} = \frac{eAD_p p_{n_0}}{L_p} \exp \left( \frac{eV_0}{kT} \right)
\]

(8.94)

then Equation (8.93) becomes

\[
\hat{I}_p = I_{p_0} \sqrt{1 + j\omega \tau_0} \left( \frac{\hat{v}_i}{V_T} \right)
\]

(8.95)

Going through the same type of analysis for the minority carrier electrons in the p region, we obtain

\[
\hat{I}_n = I_{n_0} \sqrt{1 + j\omega \tau_0} \left( \frac{\hat{v}_i}{V_T} \right)
\]

(8.96)