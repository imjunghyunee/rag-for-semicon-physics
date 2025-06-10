# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

The minority carrier mobility, lifetime, and diffusion coefficient can be determined from this single experiment. As a good first approximation, the peak of the minority carrier pulse will arrive at contact B when the experiment involving distance and time in Equation (6.70) is zero, or

\[
x - \mu_n E_0 t = 0
\]

(6.79a)

In this case \( x = d \), where \( d \) is the distance between contacts A and B, and \( t = t_0 \), where \( t_0 \) is the time at which the peak of the pulse reaches contact B. The mobility may be calculated as

\[
\mu_n = \frac{d}{E_0 t_0}
\]

(6.79b)

Figure 6.13 again shows the output response as a function of time. At times \( t_1 \) and \( t_2 \), the magnitude of the excess concentration is \( e^{-1} \) of its peak value. If the time difference between \( t_1 \) and \( t_2 \) is not too large, \( e^{-t/\tau_p} \) and \( (4\pi D_p t)^{1/2} \) do not change appreciably during this time; then the equation

\[
(d - \mu_n E_0 t) = 4D_p t
\]

(6.80)

is satisfied at both \( t = t_1 \) and \( t = t_2 \). If we set \( t = t_1 \) and \( t = t_2 \) in Equation (6.80) and add the two resulting equations, we may show that the diffusion coefficient is given by

\[
D_p = \frac{(\mu_n E_0)^2 (\Delta t)^2}{16 t_0}
\]

(6.81)

where

\[
\Delta t = t_2 - t_1
\]

(6.82)

The area \( S \) under the curve shown in Figure 6.13 is proportional to the number of excess holes that have not recombined with majority carrier electrons. We may write

\[
S = K \exp\left(-\frac{t_0}{\tau_p}\right) = K \exp\left(-\frac{d}{\mu_n E_0 \tau_p}\right)
\]

(6.83)

where \( K \) is a constant. By varying the electric field, the area under the curve will change. A plot of \(\ln(S)\) as a function of \((d/\mu_n E_0)\) will yield a straight line whose slope is \((1/\tau_p)\), so the minority carrier lifetime can also be determined from this experiment.

!Figure 6.13

**Figure 6.13** | The output excess carrier pulse versus time to determine the diffusion coefficient.