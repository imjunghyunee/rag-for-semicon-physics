# 12.3 Transistor Currents and Low-Frequency Common-Base Current Gain

The currents \( J_{RB}, J_{E\ell}, \) and \( J_R \) are B–E junction currents only and do not contribute to the collector current. The currents \( J_{x0} \) and \( J_G \) are B–C junction currents only. These current components do not contribute to the transistor action or the current gain.

The dc common-base current gain is defined as

\[
\alpha_0 = \frac{I_C}{I_E}
\]

(12.27)

If we assume that the active cross-sectional area is the same for the collector and emitter, then we can write the current gain in terms of the current densities, or

\[
\alpha_0 = \frac{J_C}{J_E} = \frac{J_{nc} + J_G + J_{x0}}{J_{xE} + J_R + J_{E\ell}}
\]

(12.28)

We are primarily interested in determining how the collector current will change with a change in emitter current. The small-signal, or sinusoidal, common-base current gain is defined as

\[
\alpha = \frac{dJ_C}{dJ_E} = \frac{J_{nc}}{J_{xE} + J_R + J_{E\ell}}
\]

(12.29)

The reverse-biased B–C currents, \( J_G \) and \( J_{x0} \), are not functions of the emitter current. We can rewrite Equation (12.29) in the form

\[
\alpha = \left( \frac{J_{xE}}{J_{xE} + J_{E\ell}} \right) \left( \frac{J_{nc}}{J_{nc} + J_R} \right) \left( \frac{J_{nc} + J_R}{J_{xE} + J_R + J_{E\ell}} \right)
\]

(12.30a)

or

\[
\alpha = \gamma \alpha_T \delta
\]

(12.30b)

The factors in Equation (12.30b) are defined as:

\[
\gamma = \left( \frac{J_{xE}}{J_{xE} + J_{E\ell}} \right) \quad \text{= emitter injection efficiency factor}
\]

(12.31a)

\[
\alpha_T = \left( \frac{J_{nc}}{J_{nc} + J_R} \right) \quad \text{= base transport factor}
\]

(12.31b)

\[
\delta = \frac{J_{nc} + J_R}{J_{xE} + J_R + J_{E\ell}} \quad \text{= recombination factor}
\]

(12.31c)

We would like to have the change in collector current be exactly the same as the change in emitter current or, ideally, to have \( \alpha = 1 \). However, a consideration of Equation (12.29) shows that \( \alpha \) will always be less than unity. The goal is to make \( \alpha \) as close to unity as possible. To achieve this goal, we must make each term in Equation (12.30b) as close to unity as possible, since each factor is less than unity.

The **emitter injection efficiency factor** \( \gamma \) takes into account the minority carrier hole diffusion current in the emitter. This current is part of the emitter current, but does not contribute to the transistor action in that \( J_{E\ell} \) is not part of the collector current. The **base transport factor** \( \alpha_T \) takes into account any recombination of excess minority carrier electrons in the base. Ideally, we want no recombination in the base. The **recombination factor** \( \delta \) takes into account the recombination in the base.