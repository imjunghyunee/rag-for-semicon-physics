# Chapter 12: The Bipolar Transistor

**inverse active**, occurs when the B–E junction is reverse biased and the B–C junction is forward biased. In this case, the transistor is operating “upside down,” and the roles of the emitter and collector are reversed. We have argued that the transistor is not a symmetrical device; therefore, the inverse-active characteristics will not be the same as the forward-active characteristics.

The junction voltage conditions for the four operating modes are shown in Figure 12.10.

## 12.1.4 Amplification with Bipolar Transistors

Voltages and currents can be amplified by bipolar transistors in conjunction with other elements. We demonstrate this amplification qualitatively in the following discussion. Figure 12.11 shows an npn bipolar transistor in a common-emitter configuration. The dc voltage sources, \( V_{BB} \) and \( V_{CC} \), are used to bias the transistor in the forward-active mode. The voltage source \( v_i \) represents a time-varying input voltage (such as a signal from a satellite) that needs to be amplified.

Figure 12.12 shows the various voltages and currents that are generated in the circuit assuming that \( v_i \) is a sinusoidal voltage. The sinusoidal voltage \( v_i \) induces a sinusoidal component of base current superimposed on a dc quiescent value. Since \( i_C = \beta i_B \), then a relatively large sinusoidal collector current is superimposed on a dc value of collector current. The time-varying collector current induces a time-varying voltage across the \( R_C \) resistor which, by Kirchhoff’s voltage law, means that a sinusoidal voltage, superimposed on a dc value, exists between the collector and emitter of the bipolar transistor. The sinusoidal voltages in the collector–emitter portion of the circuit are larger than the signal input voltage \( v_i \), so that the circuit has produced a **voltage gain** in the time-varying signals. Hence, the circuit is known as a **voltage amplifier**.

### Figures

**Figure 12.10** | Junction voltage conditions for the four operating modes of a bipolar transistor.

|            | \( V_{CE} \) |            |
|------------|--------------|------------|
| Cutoff     | Forward      |            |
|            | active       |            |
| Inverse    | Saturation   |            |
| active     |              |            |
|            | \( V_{BE} \) |            |

**Figure 12.11** | Common-emitter npn bipolar circuit configuration with a time-varying signal voltage \( v_i \) included in the base–emitter loop.

- Circuit diagram showing:
  - \( R_B \)
  - \( i_B \)
  - \( v_i \)
  - \( V_{BB} \)
  - \( i_C \)
  - \( v_R \)
  - \( R_C \)
  - \( v_{CE} \)
  - \( V_{CC} \)