# 12.4 Nonideal Effects

!Figure 12.22  
*The collector current versus collector–emitter voltage showing the Early effect and Early voltage.*

From Figure 12.22, we can write that

\[
\frac{dI_C}{dV_{CE}} = g_o = \frac{I_C}{V_{CE} + V_A} = \frac{1}{r_o}
\]

(12.45a)

where \( V_A \) and \( V_{CE} \) are defined as positive quantities, \( g_o \) is defined as the output conductance, and \( r_o \) is defined as the output resistance. Equation (12.45a) can be rewritten in the form

\[
I_C = g_o (V_{CE} + V_A) = \frac{1}{r_o} (V_{CE} + V_A)
\]

(12.45b)

showing that the collector current is now an explicit function of the collector–emitter voltage or the collector–base voltage.

----

**Objective:** Calculate the change in collector current with a change in neutral base width, and estimate the Early voltage.

**Example 12.8**

Consider a uniformly doped silicon pnp bipolar transistor with the following parameters:  
\( N_B = 5 \times 10^{16} \, \text{cm}^{-3} \), \( N_C = 2 \times 10^{15} \, \text{cm}^{-3} \), \( x_{B0} = 0.70 \, \mu\text{m} \), and \( D_B = 25 \, \text{cm}^2/\text{s} \). Assume that \( x_{B0} \ll L_B \) and that \( V_{BE} = 0.60 \, \text{V} \). The collector–base voltage is in the range \( 2 \leq V_{CB} \leq 10 \, \text{V} \).

### Solution

Assuming \( x_{B0} \ll L_B \), the excess minority carrier electron concentration in the base can be approximated by Equation (12.15b), which is

\[
\delta n_B(x) \approx \frac{n_{B0}}{x_B} \left[ \left( \exp\left(\frac{V_{BE}}{V_T}\right) - 1 \right) (x_B - x) - x \right]
\]

The collector current is

\[
|I_C| = e D_B \left( \frac{d(\delta n_B(x))}{dx} \right)_{x=0} = e D_B \frac{n_{B0}}{x_B} \exp\left(\frac{V_{BE}}{V_T}\right)
\]

The value of \( n_{B0} \) is found as

\[
n_{B0} = \frac{n_i^2}{N_B} = \frac{(1.5 \times 10^{10})^2}{5 \times 10^{16}} = 4.5 \times 10^3 \, \text{cm}^{-3}
\]

For \( V_{CB} = 2 \, \text{V} \), we find (see the following Exercise Problem Ex 12.8)

\[
x_B = x_{B0} - x_{BB} = 0.70 - 0.0518 = 0.6482 \, \mu\text{m}
\]