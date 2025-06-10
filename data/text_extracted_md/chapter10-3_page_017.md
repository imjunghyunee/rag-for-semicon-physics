# 10.5 The CMOS Technology

In the ideal MOSFET, the overlap or parasitic capacitances, \( C_{\text{gs}} \) and \( C_{\text{gd}} \), are zero. Also, when the transistor is biased in the saturation region, \( C_{\text{gd}} \) approaches zero and \( C_{\text{gs}} \) is approximately \( C_{\text{ox}}WL \). The transconductance of the ideal MOSFET biased in the saturation region and assuming a constant mobility is given by Equation (10.77) as

\[
g_m = \frac{W \mu_n C_{\text{ox}}}{L} (V_{\text{GS}} - V_T)
\]

Then, for this ideal case, the cutoff frequency is

\[
f_T = \frac{g_m}{2 \pi C_G} = \frac{W \mu_n C_{\text{ox}}}{L} \frac{(V_{\text{GS}} - V_T)}{2 \pi (C_{\text{ox}}WL)} = \frac{\mu_n (V_{\text{GS}} - V_T)}{2 \pi L^2}
\]

(10.96)

## Objective

Calculate the cutoff frequency of an ideal MOSFET with a constant mobility.

**Solution**

From Equation (10.96), the cutoff frequency is

\[
f_T = \frac{\mu_n (V_{\text{GS}} - V_T)}{2 \pi L^2} = \frac{400 \times (3 - 1)}{2 \pi \times (4 \times 10^{-5})^2} = 796 \text{ MHz}
\]

**Comment**

In an actual MOSFET, the effect of the parasitic capacitance will substantially reduce the cutoff frequency from that calculated in this example.

## EXERCISE PROBLEM

**Ex 10.10** An n-channel silicon MOSFET has the following parameters: \( \mu_n = 420 \text{ cm}^2/\text{V·s}, t_{\text{ox}} = 18 \text{ nm} = 180 \text{ Å}, L = 1.2 \mu\text{m}, W = 24 \mu\text{m}, \) and \( V_T = 0.4 \text{ V} \). The transistor is biased in the saturation region at \( V_{\text{GS}} = 1.5 \text{ V} \). Determine the cutoff frequency.

## TEST YOUR UNDERSTANDING

**TYU 10.10** Consider the n-channel MOSFET described in Exercise Problem Ex 10.10. The transistor is connected to an effective load resistance of \( R_L = 100 \text{ k}\Omega \). Calculate the ratio of Miller capacitance \( C_{\text{m}} \) to gate-to-drain capacitance \( C_{\text{gd}} \).

----

# *10.5.1 THE CMOS TECHNOLOGY

The primary objective of this book is to present the basic physics of semiconductor materials and devices without considering in detail the various fabrication processes; this important subject is left to other books. However, there is one MOS technology...