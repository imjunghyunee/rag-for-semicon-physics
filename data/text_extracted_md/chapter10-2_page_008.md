# CHAPTER 10 Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

and

\[
x_{T} = \left[ \frac{4 \varepsilon_s \phi_b}{e N_a} \right]^{1/2} = \left[ \frac{(4 \times 11.7) \times 8.85 \times 10^{-14} \times 0.3473}{(1.6 \times 10^{-19}) \times (10^{16})} \right]^{1/2}
\]

\[
\approx 0.30 \times 10^{-4} \text{ cm}
\]

Then

\[
C'_{min} = \frac{\varepsilon_{ox}}{t_{ox} + \left( \frac{\varepsilon_{s}}{\varepsilon_{i}} \right) x_{T}} = \frac{(3.9) \times (8.85 \times 10^{-14})}{180 \times 10^{-8} + \left( \frac{3.9}{11.7} \right) \times 0.30 \times 10^{-4}}
\]

\[
= 2.925 \times 10^{-8} \text{ F/cm}^2
\]

We may note that

\[
\frac{C'_{min}}{C_{ox}} = \frac{2.925 \times 10^{-8}}{1.9175 \times 10^{-7}} = 0.1525
\]

The flat-band capacitance is

\[
C'_{FB} = \frac{\varepsilon_{ox}}{t_{ox} + \left( \frac{\varepsilon_{s}}{\varepsilon_{i}} \right) \sqrt{\frac{V_{t} \varepsilon_{s}}{e N_a}}}
\]

\[
= \frac{(3.9) \times (8.85 \times 10^{-14})}{180 \times 10^{-8} + \left( \frac{3.9}{11.7} \right) \sqrt{\frac{(0.0259) \times (11.7) \times 8.85 \times 10^{-14}}{(1.6 \times 10^{-19}) \times (10^{16})}}}
\]

\[
= 1.091 \times 10^{-7} \text{ F/cm}^2
\]

We also note that

\[
\frac{C'_{FB}}{C_{ox}} = \frac{1.091 \times 10^{-7}}{1.9175 \times 10^{-7}} = 0.569
\]

**Comment**

The ratios of \( C'_{min}/C_{ox} \) and \( C'_{FB}/C_{ox} \) are typical values obtained in C-V plots.

**EXERCISE PROBLEM**

Ex 10.6 Consider a MOS capacitor with the following parameters: n\(^+\) polysilicon gate, \( N_a = 3 \times 10^{16} \text{ cm}^{-3} \), \( t_{ox} = 8 \text{ nm} = 80 \text{ Å} \), and \( Q_i' = 2 \times 10^{10} \text{ cm}^{-2} \). Determine the ratios \( C'_{min}/C_{ox} \) and \( C'_{FB}/C_{ox} \).

If we assume the oxide capacitance per unit area is \( C_{ox} = 1.9175 \times 10^{-7} \text{ F/cm}^2 \) and the channel length and width are \( L = 2 \, \mu\text{m} \) and \( W = 20 \, \mu\text{m} \), respectively, then the total gate oxide capacitance is

\[
C_{oxT} = C_{ox} \cdot L \cdot W = (1.9175 \times 10^{-7}) \times (2 \times 10^{-4}) \times (20 \times 10^{-4})
\]

\[
= 7.67 \times 10^{-14} \text{ F} = 0.0767 \text{ pF} = 76.7 \text{ fF}
\]

The total oxide capacitance in a typical MOS device is quite small.