# Chapter 8: The pn Junction Diode

n region. This distribution yields the ideal hole diffusion current density and is a function of the minority carrier hole diffusion length and the applied junction voltage. The distribution is established as a result of holes being injected across the space charge region. If, now, some of the injected holes in the space charge region are lost due to recombination, then additional holes must be injected from the p region to make up for this loss. The flow of these additional injected carriers, per unit time, results in the recombination current. This added component is schematically shown in the figure.

The total forward-bias current density is the sum of the recombination and the ideal diffusion current densities, so we can write

\[
J = J_{\text{rec}} + J_D
\]

(8.56)

where \( J_{\text{rec}} \) is given by Equation (8.55) and \( J_D \) is given by

\[
J_D = J_s \exp\left(\frac{eV_a}{kT}\right)
\]

(8.57)

The (−1) term in Equation (8.27) has been neglected. The parameter \( J_s \) is the ideal reverse-saturation current density, and from previous discussion, the value of \( J_{a} \) from the recombination current is larger than the value of \( J_s \).

If we take the natural log of Equations (8.55) and (8.57), we obtain

\[
\ln J_{\text{rec}} = \ln J_{a} + \frac{eV_a}{2kT} = \ln J_{a} + \frac{V_a}{2V_t}
\]

(8.58a)

and

\[
\ln J_D = \ln J_s + \frac{eV_a}{kT} = \ln J_s + \frac{V_a}{V_t}
\]

(8.58b)

Figure 8.16 shows the recombination and diffusion current components plotted on a log current scale as a function of \( V_a/V_t \). The slopes of the two curves are not the same. Also shown in the figure is the total current density—the sum of the two current components. We may notice that, at a low current density, the recombination current dominates, and at a higher current density, the ideal diffusion current dominates.

In general, the diode current–voltage relationship may be written as

\[
I = I_s \left[\exp\left(\frac{eV_a}{nkT}\right) - 1\right]
\]

(8.59)

where the parameter \( n \) is called the **ideality factor**. For a large forward-bias voltage, \( n \approx 1 \) when diffusion dominates, and for low forward-bias voltage, \( n \approx 2 \) when recombination dominates. There is a transition region where \( 1 < n < 2 \).

## 8.2.2 High-Level Injection

In the derivation of the ideal diode \( I-V \) relationship, we assumed that low injection was valid. Low injection implies that the excess minority carrier concentrations are always much less than the majority carrier concentration.