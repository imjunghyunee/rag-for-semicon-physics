```markdown
# CHAPTER 15 Semiconductor Microwave and Power Devices

**EXAMPLE 15.2**

**Objective:** Find the optimum drain resistor in a MOSFET inverter circuit.

A MOSFET inverter circuit is shown in Figure 15.26. Two different MOSFETs are being considered for use in the circuit. The parameters for devices A and B are given.

| Device A       | Device B       |
|----------------|----------------|
| \( BV_{\text{DSS}} = 35 \, \text{V} \) | \( BV_{\text{DSS}} = 35 \, \text{V} \) |
| \( P_T = 30 \, \text{W} \)     | \( P_T = 30 \, \text{W} \)     |
| \( I_{D, \text{max}} = 6 \, \text{A} \) | \( I_{D, \text{max}} = 4 \, \text{A} \) |

**Solution**

The SOA curves for the two devices are shown in Figure 15.27.

The load line for the inverter circuit using device A is shown as curve A. The load line intersects the voltage axis at \( V_{DD} = 24 \, \text{V} \). This curve is tangent to the maximum power curve and intersects the current axis at \( I_D = 5 \, \text{A} \). Note that, if we had wanted the load line to intersect the maximum rated current of \( I_{D, \text{max}} = 6 \, \text{A} \), the load line would have gone outside of the SOA.

For the load line A, the drain resistance is

\[
R_D = \frac{V_{DD}}{I_D} = \frac{24}{5} = 4.8 \, \Omega
\]

!Figure 15.26 | A MOSFET inverter circuit.

!Figure 15.27 | Safe operating area and load lines for devices in Example 15.2.

- **Figure 15.27** shows the safe operating area and load lines for devices in Example 15.2.
- The graph plots \( I_D \) (A) against \( V_{DS} \) (V).
- Key points include:
  - \( I_{D, \text{max}} = 6 \, \text{A} \)
  - \( I_{D, \text{max}} = 4 \, \text{A} \)
  - Maximum power dissipated
  - \( V_{DD} = 24 \, \text{V} \)
  - \( BV_{\text{DSS}} = 35 \, \text{V} \)
```