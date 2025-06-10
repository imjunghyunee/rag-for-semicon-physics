# 12.4 Nonideal Effects

!Figure 12.35  
**Figure 12.35** | Relative breakdown voltages and saturation currents of the open-base and open-emitter configurations.

The condition for breakdown corresponds to

\[
\alpha M = 1
\]

(12.59)

Using Equation (12.56) and assuming that \( V_{CB} \approx V_{CE} \), Equation (12.59) becomes

\[
\frac{\alpha}{1 - (BV_{CBO}/BV_{CEO})^n} = 1
\]

(12.60)

where \( BV_{CEO} \) is the C–E voltage at breakdown in the open-base configuration. Solving for \( BV_{CEO} \), we find

\[
BV_{CEO} = BV_{CBO} \sqrt{1 - \alpha}
\]

(12.61)

where, again, \( \alpha \) is the common-base current gain. The common-emitter and common-base current gains are related by

\[
\beta = \frac{\alpha}{1 - \alpha}
\]

(12.62a)

Normally \( \alpha \approx 1 \), so that

\[
1 - \alpha \approx \frac{1}{\beta}
\]

(12.62b)

Then Equation (12.61) can be written as

\[
BV_{CEO} = \frac{BV_{CBO}}{\sqrt{\beta}}
\]

(12.63)

The breakdown voltage in the open-base configuration is smaller, by the factor \(\sqrt{\beta}\), than the actual avalanche junction breakdown voltage. This characteristic is shown in Figure 12.35.

----

**Objective:** Design a bipolar transistor to meet a breakdown voltage specification.

**Design Example 12.11**

Consider a silicon bipolar transistor with a common-emitter current gain of \(\beta = 100\) and a base doping concentration of \(N_s = 10^{17} \text{ cm}^{-3}\). The minimum open-base breakdown voltage is to be 15 V.