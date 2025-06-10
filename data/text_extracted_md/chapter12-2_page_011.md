## 12.3 Transistor Currents and Low-Frequency Common-Base Current Gain

The base transport factor, from Equation (12.39a), is

\[
\alpha_T = \frac{1}{\cosh \left( \frac{0.70 \times 10^{-4}}{3.54 \times 10^{-5}} \right)} = 0.9998
\]

The recombination factor, from Equation (12.44), is

\[
\delta = \frac{1}{1 + \frac{5 \times 10^{-8}}{J_{0}} \exp \left( \frac{-0.65}{2(0.0259)} \right)}
\]

where

\[
J_{0} = \frac{e D_p n_{bo}}{L_{b} \tanh \left( \frac{L_{b}}{L_{b}} \right)} = \frac{(1.6 \times 10^{-19})(25)(2.25 \times 10^{15})}{3.54 \times 10^{-4} \tanh (1.977 \times 10^{-3})} = 1.29 \times 10^{-7} \, \text{A/cm}^2
\]

We can now calculate \(\delta = 0.99986\). The common-base current gain is then

\[
\alpha = \gamma \alpha_T \delta = (0.9944)(0.9998)(0.99986) = 0.99406
\]

which gives a common-emitter current gain of

\[
\beta = \frac{\alpha}{1 - \alpha} = \frac{0.99406}{1 - 0.99406} = 167
\]

### Comment

In this example, the emitter injection efficiency is the limiting factor in the current gain.

### EXERCISE PROBLEM

**Ex 12.7** Assume that \(\gamma = \alpha_T = 0.9980\), \(J_0 = 5 \times 10^{-8} \, \text{A/cm}^2\), and \(J_0 = 2 \times 10^{-11} \, \text{A/cm}^2\). Determine the common-emitter current gain \(\beta\) for (a) \(V_{BE} = 0.550 \, \text{V}\) and (b) \(V_{BE} = 0.650 \, \text{V}\).

----

### TEST YOUR UNDERSTANDING

**NOTE:** In the following Test Your Understanding questions, assume a silicon npn bipolar transistor at \(T = 300 \, \text{K}\) has the following minority carrier parameters: \(D_e = 8 \, \text{cm}^2/\text{s}\), \(D_B = 20 \, \text{m}^2/\text{s}\), \(D_C = 12 \, \text{cm}^2/\text{s}\), \(\tau_B = 10^{-8} \, \text{s}\), \(\tau_{BO} = 10^{-7} \, \text{s}\), and \(\tau_{CO} = 10^{-6} \, \text{s}\).

**TYU 12.4** If the emitter doping concentration is \(N_E = 5 \times 10^{18} \, \text{cm}^{-3}\), find the base doping concentration such that the emitter injection efficiency is \(\gamma = 0.9950\). Assume \(x_E = 2x_B = 2 \, \mu\text{m}\).

**TYU 12.5** Assume that \(\alpha_T = \delta = 0.9967\), \(x_B = x_E = 1 \, \mu\text{m}\), \(N_B = 5 \times 10^{16} \, \text{cm}^{-3}\), and \(N_E = 5 \times 10^{18} \, \text{cm}^{-3}\). Determine the common-emitter current gain \(\beta\).

**TYU 12.6** Assume that \(\gamma = \delta = 0.9967\) and \(x_B = 0.80 \, \mu\text{m}\). Determine the common-emitter current gain \(\beta\).