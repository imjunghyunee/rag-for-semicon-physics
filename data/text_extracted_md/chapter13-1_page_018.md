# CHAPTER 13 The Junction Field-Effect Transistor

The ideal drain current in the saturation region for the JFET is given by Equation (13.35). The transconductance in the saturation region is then found to be

\[
g_m = \frac{\partial I_D(\text{sat})}{\partial V_{GS}} = \frac{3I_{DSS}}{V_p} \left( 1 - \sqrt{\frac{V_{GS} - V_{GS}}{V_p}} \right) = G_{0} \left( 1 - \sqrt{\frac{V_{GS} - V_{GS}}{V_p}} \right)
\]

(13.41a)

Using the current–voltage approximation given by Equation (13.14), we can also write the transconductance as

\[
g_m = -\frac{2I_{DSS}}{V_p} \left( 1 - \frac{V_{GS}}{V_p} \right)
\]

(13.41b)

Since \( V_p \) is negative for the n-channel JFET, \( g_m \) is positive.

## EXAMPLE 13.4

**Objective:** Determine the maximum transconductance of an n-channel depletion mode JFET biased in the saturation region.

Consider the silicon JFET described in Example 13.3. We had calculated \( I_{DSS} = 0.522 \, \text{mA} \), \( V_{GS} = 0.814 \, \text{V} \), and \( V_p = 4.35 \, \text{V} \).

**Solution**

The maximum transconductance occurs when \( V_{GS} = 0 \). Then Equation (13.41a) can be written as

\[
g_m(\text{max}) = \frac{3I_{DSS}}{V_p} \left( 1 - \frac{V_{GS}}{V_p} \right) = \frac{3(0.522)}{4.35} \left( 1 - \sqrt{\frac{0.814}{4.35}} \right) = 0.204 \, \text{mA/V}
\]

**Comment**

The saturation transconductance is a function of \( V_{GS} \) and becomes zero when \( V_{GS} = V_p \).

### EXERCISE PROBLEM

**Ex 13.4** Determine the maximum transconductance of the n-channel JFET described in Exercise Problem Ex 13.3.

\[
[\Delta V_{GS} 601 = (\text{expu} = \text{exg}) \text{suV}]
\]

The experimental transconductance may deviate from this ideal expression due to a source series resistance. This effect will be considered later in the discussion of the small signal model of the JFET.

## 13.2.4 The MESFET

So far in our discussion, we have explicitly considered the pn JFET. The MESFET is the same basic device except that the pn junction is replaced by a Schottky barrier rectifying junction. The simplified MESFET geometry is shown in Figure 13.9b. MESFETs are usually fabricated in gallium arsenide. We will neglect any depletion region that may exist between the n channel and the substrate. We have also limited our discussion to depletion mode devices, wherein a gate-to-source voltage is applied to turn the transistor off. Enhancement mode GaAs MESFETs can be fabricated—their basic operation is discussed in Section 13.1.2. We can also consider enhancement mode GaAs pn JFETs.