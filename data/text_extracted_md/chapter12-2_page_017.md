### 12.4 Nonideal Effects

!Bandgap narrowing factor versus donor impurity concentration in silicon.

**Figure 12.26** | Bandgap narrowing factor versus donor impurity concentration in silicon.  
*(From Sze [19].)*

where \( E_{g0} \) is the bandgap energy at a low doping concentration and \( \Delta E_g \) is the bandgap narrowing factor.

The emitter injection efficiency factor is given by Equation (12.35) as

\[
\gamma = \frac{1}{1 + \frac{p_{E0} D_p L_B}{n_{D0} D_n L_E} \tanh(x_B/L_B) \tanh(x_E/L_E)}
\]

The term \( p_{E0} \) is the thermal-equilibrium minority carrier concentration in the emitter, taking into account bandgap narrowing, and can be written as

\[
p_{E0} = \frac{n_i^2}{N_E} = \frac{n_i^2}{N_E} \exp\left(\frac{\Delta E_g}{kT}\right) \tag{12.50}
\]

As the emitter doping concentration increases, \( \Delta E_g \) increases; thus, \( p_{E0} \) does not continue to decrease with increasing emitter doping \( N_E \). If \( p_{E0} \) starts to increase because of the bandgap narrowing, the emitter injection efficiency begins to fall off instead of continuing to increase with increased emitter doping.

----

**Objective:** Determine the increase in \( p_{E0} \) in the emitter due to bandgap narrowing.

**Example 12.9**

Consider a silicon emitter at \( T = 300 \, \text{K} \). Assume the emitter doping increases from \( 10^{18} \) to \( 10^{19} \, \text{cm}^{-3} \). Determine the new value of \( p_{E0} \) and determine the ratio \( p_{E0}/p_{B0} \).

**Solution**

For emitter doping concentrations of \( N_E = 10^{18} \) and \( 10^{19} \, \text{cm}^{-3} \), we have, neglecting bandgap narrowing,

\[
p_{E0} = \frac{n_i^2}{N_E} = \frac{(1.5 \times 10^{10})^2}{10^{18}} = 2.25 \times 10^2 \, \text{cm}^{-3}
\]

and

\[
p_{E0} = \frac{n_i^2}{N_E} = \frac{(1.5 \times 10^{10})^2}{10^{19}} = 2.25 \times 10^1 \, \text{cm}^{-3}
\]