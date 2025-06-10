# CHAPTER 9 Metal–Semiconductor and Semiconductor Heterojunctions

!Figure 9.8

**Figure 9.8** Experimental and theoretical reverse-biased currents in a PtSi-Si diode. (From Sze and Ng [15].)

\[
J_{R} = A^* T^2 \exp\left(\frac{-e\phi_{B0}}{kT}\right) \exp\left(\frac{e\Delta\phi}{kT}\right)
\]

(9.27)

The change in barrier height, \(\Delta \phi\), will increase with an increase in the electric field, or with an increase in the applied reverse-biased voltage. Figure 9.8 shows a typical reverse-biased current–voltage characteristic of a Schottky barrier diode. The reverse-biased current increases with reverse-biased voltage because of the barrier lowering effect. This figure also shows the Schottky barrier diode going into breakdown.

## EXAMPLE 9.4

**Objective:** Determine the effective Richardson constant from the current–voltage characteristics.

Consider the tungsten–silicon diode curve in Figure 9.9 and assume a barrier height of \(\phi_{B0} = 0.67 \, \text{V}\). From the figure, \(J_{R} = 6 \times 10^{-5} \, \text{A/cm}^2\).

### Solution

We have that

\[
J_{T} = A^* T^2 \exp\left(\frac{-e\phi_{B0}}{kT}\right)
\]

so that

\[
A^* = \frac{J_{T}}{T^2} \exp\left(\frac{e\phi_{B0}}{kT}\right)
\]

Then

\[
A^* = \frac{6 \times 10^{-5}}{(300)^2} \exp\left(\frac{0.67}{0.0259}\right) = 114 \, \text{A/K}^2\cdot\text{cm}^2
\]

### Comment

The experimentally determined value of \(A^*\) is a very strong function of \(\phi_{B0}\), since \(\phi_{B0}\) is in the exponential term. A small change in \(\phi_{B0}\) will change the value of the Richardson constant substantially.