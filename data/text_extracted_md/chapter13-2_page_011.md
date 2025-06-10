# 13.4 Equivalent Circuit and Frequency Limitations

When the output is short-circuited,

\[
I_t = j\omega (C_P + C_{pd})V_{gs}
\]

(13.62)

If we let \( C_G = C_P + C_{pd} \), then at the cutoff frequency

\[
|I| = 2\pi f_T C_G V_{gs} = g_m V_{gs}
\]

(13.63)

or

\[
f_T = \frac{g_m}{2\pi C_G}
\]

(13.64)

From Equation (13.41b), the maximum possible transconductance is

\[
g_m (\text{max}) = g_{0m} = \frac{e\mu_n N_d w a}{L}
\]

(13.65)

and the minimum gate capacitance is

\[
C_G (\text{min}) = \frac{\varepsilon_s WL}{a}
\]

(13.66)

where \( a \) is the maximum space charge width. The maximum cutoff frequency can be written as

\[
f_T = \frac{e\mu_n N_d a^2}{2\pi \varepsilon_s L^2}
\]

(13.67)

----

## Objective

Calculate the cutoff frequency of a silicon JFET.

Consider a silicon JFET with the following parameters:

- \( \mu_n = 1000 \, \text{cm}^2/\text{V}\cdot\text{s} \)
- \( a = 0.60 \, \mu\text{m} \)
- \( N_d = 10^{16} \, \text{cm}^{-3} \)
- \( L = 5 \, \mu\text{m} \)

## Solution

Substituting the parameters into Equation (13.67), we have

\[
f_T = \frac{e\mu_n N_d a^2}{2\pi \varepsilon_s L^2} = \frac{(1.6 \times 10^{-19})(1000)(10^{16})(0.6 \times 10^{-4})^2}{2\pi(11.7)(8.85 \times 10^{-14})(5 \times 10^{-4})^2} = 3.54 \, \text{GHz}
\]

## Comment

This example shows that even silicon JFETs can have relatively large cutoff frequencies.

## Exercise Problem

**Ex 13.9** The parameters of an n-channel silicon JFET are \( \mu_n = 1000 \, \text{cm}^2/\text{V}\cdot\text{s}, N_d = 5 \times 10^{15} \, \text{cm}^{-3}, a = 0.50 \, \mu\text{m}, \) and \( L = 2 \, \mu\text{m} \). Determine the cutoff frequency.

*(H\&S 69 L = 4f \(\mu\)N)*

For gallium arsenide JFETs or MESFETs with very small geometries, the cutoff frequency is even larger. The channel transit time may also become a factor in very high frequency devices, in which case the expression for cutoff frequency would need to be modified.