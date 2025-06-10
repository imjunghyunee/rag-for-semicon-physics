# Objective

Calculate the body-effect coefficient and the change in the threshold voltage due to an applied source-to-body voltage.

Consider an n-channel silicon MOSFET at \( T = 300 \, \text{K} \). Assume the substrate is doped to \( N_a = 3 \times 10^{16} \, \text{cm}^{-3} \) and assume the oxide is silicon dioxide with a thickness of \( t_{\text{ox}} = 20 \, \text{nm} = 200 \, \text{Å} \). Let \( V_{SB} = 1 \, \text{V} \).

# Solution

We can calculate that

\[
\phi_p = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{3 \times 10^{16}}{1.5 \times 10^{10}} \right) = 0.3758 \, \text{V}
\]

and

\[
C_{\text{ox}} = \frac{\varepsilon_{\text{ox}}}{t_{\text{ox}}} = \frac{(3.9)(8.85 \times 10^{-14})}{200 \times 10^{-8}} = 1.726 \times 10^{-7} \, \text{F/cm}^2
\]

From Equation (10.82), we find the body-effect coefficient to be

\[
\gamma = \frac{\sqrt{2q \varepsilon_s N_a}}{C_{\text{ox}}} = \frac{[2(1.6 \times 10^{-19})(11.7)(8.85 \times 10^{-14})(3 \times 10^{16})]^{1/2}}{1.726 \times 10^{-7}}
\]

or

\[
\gamma = 0.5776 \, \text{V}^{1/2}
\]

The change in threshold voltage for \( V_{SB} = 1 \, \text{V} \) is found to be

\[
\Delta V_T = \gamma \left[ \sqrt{2\phi_p + V_{SB}} - \sqrt{2\phi_p} \right]
\]

\[
= (0.5776) \left[ \sqrt{2(0.3758) + 1} - \sqrt{2(0.3758)} \right]
\]

\[
= (0.5776)[1.3235 - 0.8669] = 0.264 \, \text{V}
\]

# Comment

Figure 10.51 shows plots of \( \sqrt{I_{D(sat)}} \) versus \( V_{GS} \) for various applied values of \( V_{SB} \). The original threshold voltage is assumed to be \( V_{T0} = 0.64 \, \text{V} \).

!Figure 10.51

**Figure 10.51** | Plots of \( \sqrt{I_D} \) versus \( V_{GS} \) at several values of \( V_{SB} \) for an n-channel MOSFET.