# 13.3 Nonideal Effects

Since the effective channel length changes with \( V_{DS} \), the drain current is now a function of \( V_{DS} \). The small-signal output impedance at the drain terminal can be defined as

\[
r_{ds} = \frac{\partial V_{DS}}{\partial I_D} = \frac{\Delta V_{DS}}{\Delta I_D}
\]

(13.55)

----

## Objective

Calculate the small-signal output resistance at the drain terminal due to channel length modulation effects.

Consider an n-channel depletion mode silicon JFET with a channel doping of \( N_d = 3 \times 10^{15} \, \text{cm}^{-3} \). Calculate \( r_{ds} \) for the case when \( V_{DS} \) changes from \( V_{DS}(1) = V_{DS(sat)} + 2.0 \) to \( V_{DS}(2) = V_{DS(sat)} + 2.5 \). Assume \( L = 10 \, \mu m \) and \( I_{D0} = 4.0 \, \text{mA} \).

### Solution

We have that

\[
r_{ds} = \frac{\Delta V_{DS}}{\Delta I_D} = \frac{V_{DS}(2) - V_{DS}(1)}{I_D(2) - I_D(1)}
\]

We can calculate the change in the channel length for the two voltages as

\[
\Delta L(2) = \left[ \frac{2 \epsilon_s (V_{DS(2)} - V_{DS(sat)})}{e N_d} \right]^{1/2} = \left[ \frac{2(11.7)(8.85 \times 10^{-14})(2.5)}{(1.6 \times 10^{-19})(3 \times 10^{15})} \right]^{1/2} = 1.04 \, \mu m
\]

and

\[
\Delta L(1) = \left[ \frac{2(11.7)(8.85 \times 10^{-14})(2.0)}{(1.6 \times 10^{-19})(3 \times 10^{15})} \right]^{1/2} = 0.929 \, \mu m
\]

The drain currents are then

\[
I_D(2) = I_{D0} \left( \frac{L}{L - \frac{1}{2} \Delta L(2)} \right) = 4.0 \left( \frac{10}{9.48} \right)
\]

and

\[
I_D(1) = I_{D0} \left( \frac{L}{L - \frac{1}{2} \Delta L(1)} \right) = 4.0 \left( \frac{10}{9.54} \right)
\]

The output resistance can be calculated as

\[
r_{ds} = \frac{2.5 - 2.0}{\frac{4}{10} \left( \frac{10}{9.48} \right) - \frac{4}{10} \left( \frac{10}{9.54} \right)} = 18.9 \, \text{k}\Omega
\]

### Comment

This value of output resistance is significantly less than the ideal value of infinity.

### EXERCISE PROBLEM

**Ex 13.8** Repeat Example 13.8 if the channel doping concentration increases to \( N_d = 10^{16} \, \text{cm}^{-3} \). All other parameters remain the same.