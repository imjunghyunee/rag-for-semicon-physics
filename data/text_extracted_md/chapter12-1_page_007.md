### 12.1 The Bipolar Transistor Action

!Figure 12.7  
**Figure 12.7** Ideal bipolar transistor common-base current–voltage characteristics.

where \( i_{E2} \) involves the minority carrier hole parameters in the emitter. The total emitter current is the sum of the two components, or

\[
i_E = i_{E1} + i_{E2} = i_C + i_{E2} = I_{SE} \exp \left( \frac{V_{BE}}{V_t} \right)
\]

(12.4)

Since all current components in Equation (12.4) are functions of \(\exp(V_{BE}/V_t)\), the ratio of collector current to emitter current is a constant. We can write

\[
\frac{i_C}{i_E} = \alpha
\]

(12.5)

where \(\alpha\) is called the **common-base current gain**. By considering Equation (12.4), we see that \(i_C < i_E\) or \(\alpha < 1\). Since \(i_{E2}\) is not part of the basic transistor action, we would like this component of current to be as small as possible. We would then like the common-base current gain to be as close to unity as possible.

Referring to Figure 12.4a and Equation (12.4), note that the emitter current is an exponential function of the base–emitter voltage and the collector current is \(i_C = \alpha i_E\). To a first approximation, the collector current is independent of the base–collector voltage as long as the B–C junction is reverse biased. We can sketch the common-base transistor characteristics as shown in Figure 12.7. The bipolar transistor acts like a constant current source.

#### Base Current

As shown in Figure 12.6, the component of emitter current \(i_{E2}\) is a B–E junction current so that this current is also a component of base current shown as \(i_{B1}\). This component of base current is proportional to \(\exp(V_{BE}/V_t)\).

There is also a second component of base current. We have considered the ideal case in which there is no recombination of minority carrier electrons with majority carrier holes in the base. However, in reality, there will be some recombination. Since majority carrier holes in the base are disappearing, they must be resupplied by a flow of positive charge into the base terminal. This flow of charge is indicated as a current \(i_{B2}\) in Figure 12.6. The number of holes per unit time recombining in the base is directly related to the number of minority carrier electrons in the base.