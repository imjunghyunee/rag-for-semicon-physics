# 14.5 Light Emitting Diodes

When a voltage is applied across a pn junction, electrons and holes are injected across the space charge region where they become excess minority carriers. These excess minority carriers diffuse into the neutral semiconductor regions where they recombine with majority carriers. If this recombination process is a direct band-to-band process, photons are emitted. The diode diffusion current is directly proportional to the recombination rate, so the output photon intensity will also be proportional to the ideal diode diffusion current. In gallium arsenide, electroluminescence originates primarily on the p side of the junction because the efficiency for electron injection is higher than that for hole injection.

## 14.5.2 Internal Quantum Efficiency

The **internal quantum efficiency** of an LED is the fraction of diode current that produces luminescence. The internal quantum efficiency is a function of the injection efficiency and a function of the percentage of radiative recombination events compared with the total number of recombination events.

The three current components in a forward-biased diode are the minority carrier electron diffusion current, the minority carrier hole diffusion current, and the space charge recombination current. These current densities can be written, respectively, as

\[
J_n = \frac{eD_n n_{p0}}{L_n} \left[ \exp\left(\frac{eV}{kT}\right) - 1 \right] \tag{14.54a}
\]

\[
J_p = \frac{eD_p p_{n0}}{L_p} \left[ \exp\left(\frac{eV}{kT}\right) - 1 \right] \tag{14.54b}
\]

and

\[
J_r = \frac{en_i W}{2\tau_0} \left[ \exp\left(\frac{eV}{2kT}\right) - 1 \right] \tag{14.54c}
\]

The recombination of electrons and holes within the space charge region is, in general, through traps near midgap and is a nonradiative process. Since luminescence is due primarily to the recombination of minority carrier electrons in GaAs, we can define an injection efficiency as the fraction of electron current to total current. Then

\[
\gamma = \frac{J_n}{J_n + J_p + J_r} \tag{14.55}
\]

where \(\gamma\) is the injection efficiency. We can make \(\gamma\) approach unity by using an n\(^+\)p diode so that \(J_p\) is a small fraction of the diode current and by forward biasing the diode sufficiently so that \(J_r\) is a small fraction of the total diode current.

Once the electrons are injected into the p region, not all electrons will recombine radiatively. We can define the radiative and nonradiative recombination rates as

\[
R_r = \frac{\delta n}{\tau_r} \tag{14.56a}
\]