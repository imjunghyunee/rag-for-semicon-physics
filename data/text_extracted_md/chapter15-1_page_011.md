# Semiconductor Microwave and Power Devices

## Safe Operating Area (SOA) of a Bipolar Transistor

### Figure 15.14

The safe operating area (SOA) of a bipolar transistor plotted on:

- (a) Linear scales
- (b) Logarithmic scales

#### (a) Linear Scales

- **Axes:**
  - \( I_C \) (A)
  - \( V_{CE} \) (V)

- **Key Points:**
  - \( I_{C,\text{max}} \)
  - \( P_T \)
  - Second breakdown
  - \( V_{CE,\text{max}} \)

#### (b) Logarithmic Scales

- **Axes:**
  - \( I_C \) (A)
  - \( V_{CE} \) (V)

- **Key Points:**
  - Maximum current limit
  - \( P_T \)
  - Second breakdown
  - \( V_{CE,\text{max}} \)

### Explanation

When the transistor is biased in the forward-active mode, the collector current begins to increase significantly before the actual breakdown voltage is reached. All the curves tend to merge to the same collector–emitter voltage once breakdown has occurred. This voltage, \( V_{CE,\text{max}} \), is the minimum voltage necessary to sustain the transistor in breakdown.

Another breakdown effect is called **second breakdown**, which occurs in a BJT operating at high voltage and high current. Slight nonuniformities in current density produce local regions of increased heating that increases the minority carrier concentrations in the semiconductor material, which in turn increases the current in these regions. This effect results in positive feedback, and the current continues to increase, producing a further increase in temperature, until the semiconductor material may actually melt, creating a short circuit between the collector and emitter.

The average power dissipated in a BJT must be kept below a specified maximum value, to ensure that the temperature of the device remains below a maximum value. If we assume the collector current and collector–emitter voltage are dc values, then at the maximum rated power \( P_T \) for the transistor, we can write:

\[
P_T = V_{CE} I_C
\]

Equation (15.6) neglects the \( V_{BE} I_B \) component of power dissipation in the transistor.

The maximum current, voltage, and power limitations can be illustrated on the \( I_C \) versus \( V_{CE} \) characteristics as shown in Figure 15.14. The average power limitation, \( P_T \), is a hyperbola described by Equation (15.6). The region where the transistor can be operated safely is known as the safe operating area (SOA) and is bounded by \( I_{C,\text{max}} \), \( V_{CE,\text{max}} \), \( P_T \), and the transistor’s second breakdown characteristic curve. Figure 15.14a shows the safe operating area using linear current and voltage scales. Figure 15.14b shows the same characteristics using log scales.