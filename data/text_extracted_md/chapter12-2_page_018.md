```markdown
# CHAPTER 12 The Bipolar Transistor

Taking into account the bandgap narrowing effect shown in Figure 12.26, we obtain, respectively, for \( N_E = 10^{18} \) and \( 10^{19} \) cm\(^{-3}\),

\[
p_{E0} = \frac{n_i^2}{N_E} \exp \left( \frac{\Delta E_g}{kT} \right) = \left( \frac{1.5 \times 10^{10}^2}{10^{18}} \right) \exp \left( \frac{0.020}{0.0259} \right) = 4.87 \times 10^2 \text{ cm}^{-3}
\]

and

\[
p_{E0} = \left( \frac{1.5 \times 10^{10}^2}{10^{19}} \right) \exp \left( \frac{0.080}{0.0259} \right) = 4.94 \times 10^1 \text{ cm}^{-3}
\]

Taking the ratio of \( p_{E0}/p_{E0} \) for \( N_E = 10^{18} \) cm\(^{-3}\), we find

\[
\frac{p_{E0}}{p_{E0}} = \exp \left( \frac{\Delta E_g}{kT} \right) = \exp \left( \frac{0.020}{0.0259} \right) = 2.16
\]

and for \( N_E = 10^{19} \) cm\(^{-3}\), we find

\[
\frac{p_{E0}}{p_{E0}} = \exp \left( \frac{0.080}{0.0259} \right) = 21.95
\]

### Comment

If the emitter doping concentration increases from \( 10^{18} \) to \( 10^{19} \) cm\(^{-3}\), the thermal-equilibrium minority carrier concentration actually increases rather than decreasing by a factor of 10 as would be expected.

### EXERCISE PROBLEM

**Ex 12.9** Determine the thermal-equilibrium minority carrier concentration for an emitter doping concentration of \( N_E = 10^{20} \) cm\(^{-3} \) taking into account bandgap narrowing.

----

As the emitter doping increases, the bandgap narrowing factor, \( \Delta E_g \), will increase; this can actually cause \( p_{E0} \) to increase. As \( p_{E0} \) increases, the emitter injection efficiency decreases; this then causes the transistor gain to decrease, as shown in Figure 12.24. A very high emitter doping may result in a smaller current gain than we anticipate because of the bandgap narrowing effect.

## 12.4.4 Current Crowding

It is tempting to neglect the effects of base current in a transistor since the base current is usually much smaller than either the collector or the emitter current. Figure 12.27 is a cross section of an npn transistor showing the lateral distribution of base current. The base region is typically less than a micrometer thick, so there can be a sizable base resistance. The nonzero base resistance results in a lateral potential difference under the emitter region. For the npn transistor, the potential decreases from the edge of the emitter toward the center. The emitter is highly doped, so as a first approximation the emitter can be considered an equipotential region.

The number of electrons from the emitter injected into the base is exponentially dependent on the B–E voltage. With the lateral voltage drop in the base between the edge and center of the emitter, more electrons will be injected near the emitter edges than in the center, causing the emitter current to be crowded toward the edges. This
```