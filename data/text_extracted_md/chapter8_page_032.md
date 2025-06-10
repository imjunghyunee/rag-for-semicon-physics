### 8.3 Small-Signal Model of the pn Junction

superimposed on the dc voltage so that the total forward-biased voltage can be written as \( V_a = V_{dc} + v(t) \) on set.

As the voltage across the junction changes, the number of holes injected across the space charge region into the n region also changes. Figure 8.19b shows the hole concentration at the space charge edge as a function of time. At \( t = t_0 \), the ac voltage is zero so that the concentration of holes at \( x = 0 \) is just given by \( p_n(0) = p_{n0} \exp(V_a/V_T) \), which is what we have seen previously.

Now, as the ac voltage increases during its positive half cycle, the concentration of holes at \( x = 0 \) will increase and reach a peak value at \( t = t_1 \), which corresponds to the peak value of the ac voltage. When the ac voltage is on its negative half cycle, the total voltage across the junction decreases so that the concentration of holes at \( x = 0 \) decreases. The concentration reaches a minimum value at \( t = t_2 \), which corresponds to the time that the ac voltage reaches its maximum negative value. The minority carrier hole concentration at \( x = 0 \), then, has an ac component superimposed on the dc value as indicated in Figure 8.19b.

As previously discussed, the holes at the space charge edge (\( x = 0 \)) diffuse into the n region where they recombine with the majority carrier electrons. We assume that the period of the ac voltage is large compared to the time it takes carriers to diffuse into the n region. The hole concentration as a function of distance into the n region can then be treated as a steady-state distribution. Figure 8.19c shows the steady-state hole concentrations at three different times. At \( t = t_0 \), the ac voltage is zero, so the \( t = t_0 \) curve corresponds to the hole distribution established by the dc voltage. The \( t = t_1 \) curve corresponds to the distribution established when the ac voltage has reached its peak positive value, and the \( t = t_2 \) curve corresponds to the distribution established when the ac voltage has reached its maximum negative value. The shaded areas represent the charge \( \Delta Q \) that is alternately charged and discharged during the ac voltage cycle.

Exactly the same process is occurring in the p region with the electron concentration. The mechanism of charging and discharging of holes in the n region and electrons in the p region leads to a capacitance. This capacitance is called diffusion capacitance. The physical mechanism of this diffusion capacitance is different from that of the junction capacitance discussed in the last chapter. We show that the magnitude of the diffusion capacitance in a forward-biased pn junction is usually substantially larger than the junction capacitance.

#### Mathematical Analysis

The minority carrier distribution in the pn junction will be derived for the case when a small sinusoidal voltage is superimposed on the dc junction voltage. We can then determine small signal, or ac, diffusion currents from these minority carrier functions. Figure 8.20 shows the minority carrier distribution in a pn junction when a forward-biased dc voltage is applied. The origin, \( x = 0 \), is set at the edge of the space charge region on the n side for convenience. The minority carrier hole concentration at \( x = 0 \) is given by Equation (8.7) as \( p_n(0) = p_{n0} \exp(eV_a/kT) \), where \( V_a \) is the applied voltage across the junction.

Now let

\[
V_a = V_0 + v(t)
\]

(8.69)