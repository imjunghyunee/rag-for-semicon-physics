# CHAPTER 11 Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

and

\[
\Delta V_{DS} = V_{DS} - V_{DS}(\text{sat}) = 2.5 - 0.6 = 1.9 \, \text{V}
\]

Using Equation (11.4), we determine

\[
\Delta L = \sqrt{\frac{7 \varepsilon_i}{eN_i}} \left[ \sqrt{\phi_{fp} + V_{DS}(\text{sat})} + \Delta V_{DS} - \sqrt{\phi_{fp} + V_{DS}(\text{sat})} \right]
\]

\[
= \frac{\sqrt{2 \times 11.7 \times 8.85 \times 10^{-14}}}{\sqrt{(1.6 \times 10^{-19}) \times 2 \times 10^{16}}} \left[ \sqrt{0.3653 + 0.6 + 1.9} - \sqrt{0.3653 + 0.6} \right]
\]

\[
= 1.807 \times 10^{-5} \, \text{cm}
\]

or

\[
\Delta L = 0.1807 \, \mu\text{m}
\]

Then

\[
\frac{I_D}{I_{D0}} = \frac{L}{L - \Delta L} = \frac{1}{1 - 0.1807} = 1.22
\]

> **Comment**  
> The actual drain current increases as the effective channel length decreases when the transistor is biased in the saturation region.

## EXERCISE PROBLEM

**Ex 11.1** An n-channel MOSFET has the same properties as described in Example 11.1 except for the channel length. The transistor is biased at \( V_{GS} = 0.8 \, \text{V} \) and \( V_{DS} = 2.5 \, \text{V} \). Find the minimum channel length such that the ratio of actual drain current to the ideal drain current due to channel length modulation is no larger than 1.35.  
(\( \mu_n C_{ox} = 7 \, \mu\text{A/V}^2 \))

## 11.1.3 Mobility Variation

In the derivation of the ideal \( I-V \) relationship, we explicitly assumed that the mobility was a constant. However, this assumption must be modified for two reasons. The first effect to be considered is the variation in mobility with gate voltage. The second reason for a mobility variation is that the effective carrier mobility decreases as the carrier approaches the velocity saturation limit. This effect is discussed in the next section.

The inversion layer charge is induced by a vertical electric field, which is shown in Figure 11.8 for an n-channel device. A positive gate voltage produces a force on the electrons in the inversion layer toward the surface. As the electrons travel through the channel toward the drain, they are attracted to the surface, but then are repelled by localized coulombic forces. This effect, schematically shown in Figure 11.9, is called **surface scattering**. The surface scattering effect reduces mobility. If there is a positive fixed oxide charge near the oxide-semiconductor interface, the mobility will be further reduced due to the additional coulomb interaction.