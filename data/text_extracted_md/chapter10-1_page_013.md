# 10.1 The Two-Terminal MOS Structure

We can define a potential \(\phi_{ms}\) as

\[
\phi_{ms} = \left[ \phi_m' - \left( \chi' + \frac{E_g}{2e} + \phi_b \right) \right]
\]

(10.15)

which is known as the metal–semiconductor work function difference.

----

## Objective

Determine the metal–semiconductor work function difference, \(\phi_{ms}\), for a given MOS system and semiconductor doping.

**Example 10.2**

For an aluminum–silicon dioxide junction, \(\phi_m' = 3.20 \, \text{V}\) and, for a silicon–silicon dioxide junction, \(\chi' = 3.25 \, \text{V}\). We may assume that \(E_g = 1.12 \, \text{V}\). Let the p-type doping be \(N_a = 10^{15} \, \text{cm}^{-3}\).

### Solution

For silicon at \(T = 300 \, \text{K}\), we may calculate \(\phi_b\) as

\[
\phi_b = V_t \ln \left( \frac{N_a}{n_i} \right) = (0.0259) \ln \left( \frac{10^{15}}{1.5 \times 10^{10}} \right) = 0.288 \, \text{V}
\]

Then the metal–semiconductor work function difference is

\[
\phi_{ms} = \phi_m' - \left( \chi' + \frac{E_g}{2e} + \phi_b \right) = 3.20 - (3.25 + 0.560 + 0.288)
\]

or

\[
\phi_{ms} = -0.898 \, \text{V}
\]

### Comment

The value of \(\phi_{ms}\) will become more negative as the doping of the p-type substrate increases.

----

## Exercise Problem

**Ex 10.2** Repeat Example 10.2 for a semiconductor doping of \(N_a = 10^{16} \, \text{cm}^{-3}\).

----

Degenerately doped polysilicon deposited on the oxide is also often used as the metal gate. Figure 10.14a shows the energy-band diagram of a MOS capacitor with an \(n^+\) polysilicon gate and a p-type substrate. Figure 10.14b shows the energy-band diagram.

!Energy-band diagram

**Figure 10.14** | Energy-band diagram through the MOS structure with a p-type substrate at zero gate bias for (a) an \(n^+\) polysilicon gate and (b) a \(p^+\) polysilicon gate.