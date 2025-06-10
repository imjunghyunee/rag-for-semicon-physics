# Chapter 13: The Junction Field-Effect Transistor

One application of GaAs FETs is in ultrafast digital integrated circuits. Conventional GaAs MESFET logic gates can achieve propagation delay times in the subnanosecond range. These delay times are at least comparable to, if not shorter than, fast ECL, but the power dissipation is three orders of magnitude smaller than in the ECL circuits. Enhancement mode GaAs JFETs have been used as drivers in logic circuits, and depletion mode devices may be used as loads. Propagation delay times of as low as 45 ps have been observed. Special JFET structures may be used to further increase the speed. These structures include the modulation-doped field-effect transistor, which is discussed in the following section.

## Test Your Understanding

**TVU 13.3**  
Consider a p-channel silicon JFET that has parameters \( a = 0.50 \, \mu m \), \( \mu_n = 400 \, \text{cm}^2/\text{V}\cdot\text{s} \), \( N_d = 2 \times 10^{16} \, \text{cm}^{-3} \), and \( L = 4 \, \mu m \). Calculate the cutoff frequency.  
\((2H9 \, LO2 \, = \, f^*SUV)\)

**TVU 13.4**  
An n-channel GaAs p JFET has parameters \( a = 0.50 \, \mu m \), \( L = 1 \, \mu m \), \( N_d = 3 \times 10^{15} \, \text{cm}^{-3} \), and \( \mu_n = 6500 \, \text{cm}^2/\text{V}\cdot\text{s} \). Determine the cutoff frequency.  
\((2H9 \, LO1 \, = \, f^*SUV)\)

## 13.5.1 High Electron Mobility Transistor

As frequency needs, power capacity, and low noise performance requirements increase, the gallium arsenide MESFET is pushed to its limit of design and performance. These requirements imply a very small FET with a short channel length, large saturation current, and large transconductance. These requirements are generally achieved by increasing the channel doping under the gate. In all of the devices we have considered, the channel region is in a doped layer of bulk semiconductor with the majority carriers and doping impurities in the same region. The majority carriers experience ionized impurity scattering, which reduces carrier mobility and degrades device performance.

The degradation in mobility and peak velocity in GaAs due to increased doping can be minimized by separating the majority carriers from the ionized impurities. This separation can be achieved in a heterostructure that has an abrupt discontinuity in conduction and valence bands. We considered the basic heterojunction properties in Chapter 9. Figure 13.23 shows the conduction-band energy relative to the Fermi energy of an N-AlGaAs-intrinsic GaAs heterojunction in thermal equilibrium. Thermal equilibrium is achieved when electrons from the wide-bandgap AlGaAs flow into the GaAs and are confined to the potential well. However, the electrons are free to move parallel to the heterojunction interface. In this structure, the majority carrier

!Figure 13.23  
**Figure 13.23** | Conduction-band edges for N-AlGaAs–intrinsic GaAs abrupt heterojunction.