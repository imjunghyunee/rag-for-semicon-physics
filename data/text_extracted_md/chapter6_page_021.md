# Chapter 6: Nonequilibrium Excess Carriers in Semiconductors

Equation (6.68) is normally solved using Laplace transform techniques. The solution, without going through the mathematical details, is

\[
p'(x, t) = \frac{1}{(4\pi D_p t)^{1/2}} \exp \left[ -\frac{(x - \mu_p E_0 t)^2}{4D_p t} \right]
\]

(6.69)

The total solution, from Equations (6.67) and (6.69), for the excess minority carrier hole concentration is

\[
\delta p(x, t) = \frac{e^{-t/\tau_p}}{(4\pi D_p t)^{1/2}} \exp \left[ -\frac{(x - \mu_p E_0 t)^2}{4D_p t} \right]
\]

(6.70)

**Comment**

We could show that Equation (6.70) is a solution to the partial differential equation, Equation (6.66), by direct substitution. We may also note that Equation (6.70) is not normalized.

## Exercise Problem

**Ex 6.5** Consider the result of Example 6.5. Let \( D_p = 10 \, \text{cm}^2/\text{s}, \tau_p = 10^{-7} \, \text{s}, \mu_p = 400 \, \text{cm}^2/\text{V}\cdot\text{s}, \) and \( E = 100 \, \text{V/cm}. \)

(a) Determine \( \delta p \) for \( t = 10^{-7} \, \text{s} \) at \( x = 20 \, \mu\text{m}, \) (ii) \( x = 40 \, \mu\text{m}, \) and (iii) \( x = 60 \, \mu\text{m}. \)

(b) Determine \( \delta p \) for \( x = 40 \, \mu\text{m} \) at (i) \( t = 5 \times 10^{-8} \, \text{s}, \) (ii) \( t = 10^{-7} \, \text{s}, \) and (iii) \( t = 2 \times 10^{-7} \, \text{s}. \) Compare the results to the curves shown in Figure 6.9.

----

Equation (6.70) can be plotted as a function of distance \( x \), for various times. Figure 6.8 shows such a plot for the case when the applied electric field is zero. For \( t > 0 \), the excess minority carrier holes diffuse in both the \( +x \) and \( -x \) directions. During this time, the excess majority carrier electrons, which were generated, diffuse at exactly the same rate as the holes. As time proceeds, the excess holes recombine with the excess electrons so that at \( t = \infty \) the excess hole concentration is zero. In this particular example, both diffusion and recombination processes are occurring at the same time.

Figure 6.9 shows a plot of Equation (6.70) as a function of distance \( x \) at various times for the case when the applied electric field is not zero. In this case, the pulse of excess minority carrier holes is drifting in the \( +x \) direction, which is the direction of the electric field. We still have the same diffusion and recombination processes as we had before. An important point to consider is that, with charge neutrality, \( \delta n = \delta p \) at any instant of time and at any point in space. The excess electron concentration is equal to the excess hole concentration. In this case, then, the excess electron pulse is moving in the same direction as the applied electric field even though the electrons have a negative charge. In the ambipolar transport process, the excess carriers are characterized by the minority carrier parameters. In this example, the excess carriers behave according to the minority carrier hole parameters, which include \( D_p, \mu_p, \) and \( \tau_p \). The excess majority carrier electrons are being pulled along by the excess minority carrier holes.