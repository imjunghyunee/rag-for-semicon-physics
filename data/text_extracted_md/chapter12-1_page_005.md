# 12.1 The Bipolar Transistor Action

!Figure 12.5  
**Figure 12.5** | Cross section of an npn bipolar transistor showing the injection and collection of electrons in the forward-active mode.

Figure 12.4b. The large gradient in the electron concentration means that electrons injected from the emitter will diffuse across the base region into the B–C space charge region, where the electric field will sweep the electrons into the collector. We want as many electrons as possible to reach the collector without recombining with any majority carrier holes in the base. For this reason, the width of the base needs to be small compared with the minority carrier diffusion length. If the base width is small, then the minority carrier electron concentration is a function of both the B–E and B–C junction voltages. The two junctions are close enough to be called interacting pn junctions.

Figure 12.5 shows a cross section of an npn transistor with the injection of electrons from the n-type emitter (hence the name emitter) and the collection of the electrons in the collector (hence the name collector).

## 12.1.2 Simplified Transistor Current Relation—Qualitative Discussion

We can gain a basic understanding of the operation of the transistor and the relations between the various currents and voltages by considering a simplified analysis. After this discussion, we delve into a more detailed analysis of the physics of the bipolar transistor.

The minority carrier concentrations are again shown in Figure 12.6 for an npn bipolar transistor biased in the forward-active mode. Ideally, the minority carrier electron concentration in the base is a linear function of distance, which implies no recombination. The electrons diffuse across the base and are swept into the collector by the electric field in the B–C space charge region.

### Collector Current

Assuming the ideal linear electron distribution in the base, the collector current can be written as a diffusion current given by

\[
i_C = eD_nA_{BE} \frac{dn(x)}{dx} \bigg|_{x=0} = eD_nA_{BE} \left[ \frac{n_{b0}(0) - 0}{0 - x_B} \right] = -\frac{eD_nA_{BE}}{x_B} \cdot n_{b0} \exp\left(\frac{V_{BE}}{V_t}\right)
\]

(12.1)

where \( A_{BE} \) is the cross-sectional area of the B–E junction, \( n_{b0} \) is the thermal-equilibrium electron concentration in the base, and \( V_t \) is the thermal voltage. The