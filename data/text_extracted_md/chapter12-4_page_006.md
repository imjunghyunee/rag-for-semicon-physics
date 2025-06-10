```markdown
# Base Transit Time and Emitter–Base Charging Time Effects

The decrease in bandgap energy from the base–emitter junction to the base–collector junction induces an electric field in the base that helps accelerate electrons across the p-type base region. For a bandgap narrowing of 100 meV, the induced electric field can be on the order of \(10^3\) to \(10^4\) V/cm. This electric field reduces the base transit time by approximately a factor of 2.5.

The emitter–base junction charging time constant, given by Equation (12.87), is directly proportional to the emitter diffusion resistance \(r'\). This parameter is inversely proportional to the emitter current, as seen in Equation (12.88). For a given base current, the emitter current in the SiGe-base transistor is larger, since the current gain is larger. The emitter–base junction charging time is then smaller in a SiGe-base transistor than that in a Si-base transistor.

The reduction in both the base transit time and the emitter–base charging time increases the cutoff frequency of the SiGe-base transistor. The cutoff frequency of these devices can be substantially higher than that of the Si-base device.

## 12.8.3 Heterojunction Bipolar Transistors

As mentioned previously, one of the basic limitations of the current gain in the bipolar transistor is the emitter injection efficiency. The emitter injection efficiency \(\gamma\) can be increased by reducing the value of the thermal-equilibrium minority carrier concentration \(p_{en}\) in the emitter. However, as the emitter doping increases, the bandgap narrowing effect offsets any improvement in the emitter injection efficiency. One possible solution is to use a wide-bandgap material for the emitter, which will minimize the injection of carriers from the base back into the emitter.

Figure 12.50a shows a discrete aluminum gallium arsenide (AlGaAs)/gallium arsenide (GaAs) heterojunction bipolar transistor, and Figure 12.50b shows the energy-band diagram of the n-AlGaAs emitter to p-GaAs base junction. The large potential barrier \(V_a\) limits the number of holes that will be injected back from the base into the emitter.

The intrinsic carrier concentration is a function of bandgap energy as

\[
n_i^2 \propto \exp\left(-\frac{E_g}{kT}\right)
\]

For a given emitter doping, the number of minority carrier holes injected into the emitter is reduced by a factor of

\[
\exp\left(\frac{\Delta E_g}{kT}\right)
\]

in changing from a narrow- to wide-bandgap emitter. If \(\Delta E_g = 0.30 \, \text{eV}\), for example, \(n_i^2\) would be reduced by approximately \(10^5\) at \(T = 300 \, \text{K}\). The drastic reduction in \(n_i^2\) for the wide-bandgap emitter means that the requirements of a very high emitter doping can be relaxed and a high emitter injection efficiency can still be obtained. A lower emitter doping reduces the bandgap narrowing effect.

The heterojunction GaAs bipolar transistor has the potential of being a very high-frequency device. A lower emitter doping in the wide-bandgap emitter leads to
```