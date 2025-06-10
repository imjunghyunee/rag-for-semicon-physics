# 15.5 Power MOSFETs

!Graphs

**Figure 15.24** | Typical characteristics for high-power MOSFETs at various temperatures: (a) transconductance versus drain current; (b) drain current versus gate-to-source voltage.

## Description

Power MOSFETs differ from bipolar power transistors in both operating principles and performance. The superior performance characteristics of power MOSFETs are faster switching times, no second breakdown, and stable gain and response time over a wide temperature range. Figure 15.24a shows the transconductance of the 2N6757 versus temperature. The variation with temperature of the MOSFET transconductance is less than the variation in the BJT current gain that is shown in Figure 15.12. Figure 15.24b is a plot of drain current versus gate-to-source voltage at three different temperatures. We may note that at high current, the current decreases with temperature at a constant gate-to-source voltage, providing the stability that has been discussed.

Power MOSFETs must operate in a SOA. As with power BJTs, the SOA is defined by three factors: the maximum drain current, \(I_{D,\text{max}}\), rated breakdown voltage, \(BV_{\text{DSS}}\), and the maximum power dissipation given by \(P_T = V_{DS}I_D\). The SOA is shown in Figure 15.25a in which the current and voltage are plotted on linear scales.

!Graphs

**Figure 15.25** | The safe operating area (SOA) of a MOSFET plotted on (a) linear scales and (b) logarithmic scales.

### Equations

- Maximum power dissipation: \( P_T = V_{DS}I_D \)

### Graphs

- **Figure 15.24a**: Transconductance vs. Drain Current
  - \( V_{DS} = 15 \, \text{V} \)
  - \( T_J = -55^\circ \text{C}, 25^\circ \text{C}, 125^\circ \text{C} \)
  - 80-\(\mu\)s pulse test

- **Figure 15.24b**: Drain Current vs. Gate-to-Source Voltage
  - \( V_{DS} = 15 \, \text{V} \)
  - \( T_J = -55^\circ \text{C}, 25^\circ \text{C}, 125^\circ \text{C} \)
  - 80-\(\mu\)s pulse test

- **Figure 15.25a**: SOA on Linear Scales
  - \( I_D \) vs. \( V_{DS} \)
  - \( I_{D,\text{max}} \), \( BV_{\text{DSS}} \), \( P_T \)

- **Figure 15.25b**: SOA on Logarithmic Scales
  - \( \log I_D \) vs. \( \log V_{DS} \)
  - \( \log I_{D,\text{max}} \), \( \log BV_{\text{DSS}} \), \( P_T \)