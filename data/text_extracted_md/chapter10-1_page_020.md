# CHAPTER 10 Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

Using the definition of flat-band voltage from Equation (10.25), we can also express the threshold voltage as

\[
V_{TN} = \frac{|Q'_s(\text{max})|}{C_{ox}} + V_{FB} + 2\phi_f
\]

(10.31c)

For a given semiconductor material, oxide material, and gate metal, the threshold voltage is a function of semiconductor doping, oxide charge \(Q'_s\), and oxide thickness.

## EXAMPLE 10.4

**Objective:** Calculate the threshold voltage of a MOS system using an aluminum gate.

Consider a p-type silicon substrate at \(T = 300 \, K\) doped to \(N_a = 10^{15} \, \text{cm}^{-3}\). Let \(Q'_s = 10^{10} \, \text{cm}^{-2}\), \(t_{ox} = 12 \, \text{nm} = 120 \, \text{Å}\), and assume the oxide is silicon dioxide.

### Solution

From Figure 10.16, we find \(\phi_{ms} = -0.88 \, V\). The other parameters are

\[
\phi_f = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{10^{15}}{1.5 \times 10^{10}} \right) = 0.2877 \, V
\]

and

\[
x_{dT} = \left[ \frac{4 \varepsilon_s \phi_f}{e N_a} \right]^{1/2} = \left[ \frac{4(11.7)(8.85 \times 10^{-14})(0.2877)}{(1.6 \times 10^{-19})(10^{15})} \right]^{1/2} = 8.63 \times 10^{-5} \, \text{cm}
\]

Then

\[
|Q'_s(\text{max})| = e N_a x_{dT} = (1.6 \times 10^{-19})(10^{15})(8.63 \times 10^{-5}) = 1.381 \times 10^{-8} \, \text{C/cm}^2
\]

The threshold voltage is now found to be

\[
V_{TN} = \left( \frac{|Q'_s(\text{max})| - Q'_s}{C_{ox}} \right) + \phi_{ms} + 2\phi_f
\]

\[
= \left[ (1.381 \times 10^{-8}) - (10^{10})(1.6 \times 10^{-19}) \right] \left( \frac{120 \times 10^{-8}}{(3.9)(8.85 \times 10^{-14})} \right)
\]

\[
+ (-0.88) + 2(0.2877)
\]

or

\[
V_{TN} = -0.262 \, V
\]

**Comment**

In this example, the semiconductor is fairly lightly doped, which, in conjunction with the positive charge in the oxide and the work function difference, is sufficient to induce an electron inversion layer charge even with zero applied gate voltage. This condition makes the threshold voltage negative.

### EXERCISE PROBLEM

**Ex 10.4** Determine the metal–semiconductor work function difference and the threshold voltage for a silicon MOS device at \(T = 300 \, K\) for the following parameters: p\(^+\) polysilicon gate, \(N_a = 2 \times 10^{16} \, \text{cm}^{-3}\), \(t_{ox} = 8 \, \text{nm} = 80 \, \text{Å}\), and \(Q'_s = 2 \times 10^{10} \, \text{cm}^{-2}\).

A negative threshold voltage for a p-type substrate implies a depletion mode device. A negative voltage must be applied to the gate in order to make the inversion.