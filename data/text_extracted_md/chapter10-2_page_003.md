# 10.1 The Two-Terminal MOS Structure

we require a metal–semiconductor work function difference value that is more positive than for the aluminum gate. We therefore choose a p\(^+\) polysilicon gate.

Consider a doping concentration of \(N_d = 10^{17} \, \text{cm}^{-3}\). From Figure 10.16, the metal–semiconductor work function difference is \(\phi_{ms} = +1.1 \, \text{V}\). The remaining parameters are found to be

\[
\phi_b = V_t \ln \left( \frac{N_d}{n_i} \right) = (0.0259) \ln \left( \frac{10^{17}}{1.5 \times 10^{10}} \right) = 0.407 \, \text{V}
\]

\[
x_{dm} = \left[ \frac{4 \varepsilon_s \phi_b}{e N_d} \right]^{1/2} = \left[ \frac{4(11.7)(8.85 \times 10^{-14})(0.407)}{(1.6 \times 10^{-19})(10^{17})} \right]^{1/2}
\]

\[
= 1.026 \times 10^{-5} \, \text{cm}
\]

\[
|Q_{si}(\text{max})| = e N_d x_{dm} = (1.6 \times 10^{-19})(10^{17})(1.026 \times 10^{-5})
\]

\[
= 1.642 \times 10^{-7} \, \text{C/cm}^2
\]

The threshold voltage is determined to be

\[
V_{TP} = -\left[ |Q_{si}(\text{max})| - Q_{d1} \cdot \frac{t_{ox}}{\varepsilon_{ox}} \right] + \phi_{ms} - 2\phi_b
\]

or

\[
V_{TP} = -\left[ (1.642 \times 10^{-7}) - (2 \times 10^9)(1.6 \times 10^{-19}) \cdot (120 \times 10^{-8}) \right] + \frac{1.1 - 2(0.407)}{(3.9)(8.85 \times 10^{-14})}
\]

which yields

\[
V_{TP} = -0.296 \, \text{V} \approx -0.3 \, \text{V}
\]

## Comment

The negative threshold voltage, with the n-type substrate, implies that this MOS capacitor is an enhancement mode device. The inversion layer charge is zero with zero applied gate voltage, and a negative gate voltage must be applied to induce the hole inversion charge.

## EXERCISE PROBLEM

**Ex 10.5** Consider a MOS capacitor with silicon dioxide and an n-type silicon substrate at \(T = 300 \, \text{K}\) with the following parameters: p\(^+\) polysilicon gate, \(N_d = 2 \times 10^{16} \, \text{cm}^{-3}\), \(t_{ox} = 20 \, \text{nm} = 200 \, \text{Å}\), and \(Q_{d1} = 5 \times 10^9 \, \text{cm}^{-2}\). Determine the threshold voltage. Is the capacitor an enhancement mode or depletion mode device?

----

## TEST YOUR UNDERSTANDING

**TYU 10.1** (a) Consider an oxide-to-n-type silicon junction at \(T = 300 \, \text{K}\). The impurity doping concentration in the silicon is \(N_d = 8 \times 10^{15} \, \text{cm}^{-3}\). Calculate the maximum space charge width in the silicon. (b) Repeat part (a) for a doping concentration of \(N_d = 4 \times 10^{16} \, \text{cm}^{-3}\).

[Hint: Use Eq. (9) and Table (9) values]