```markdown
The storage time \( t_s \) can be determined by solving the time-dependent continuity equation. If we consider a one-sided p\(^+\)n junction, the storage time is determined from the equation

\[
\text{erf} \left( \sqrt{\frac{t_s}{\tau_{p0}}} \right) = \frac{I_f}{I_f + I_R}
\]

(8.109)

where \(\text{erf} (x)\) is known as the error function. An approximate solution for the storage time can be obtained as

\[
t_s \approx \tau_{p0} \ln \left( 1 + \frac{I_f}{I_R} \right)
\]

(8.110)

The recovery phase for \( t > t_s \) is the time required for the junction to reach its steady-state reverse-biased condition. The remainder of the excess charge is being removed and the space charge width is increasing to the reverse-biased value. The decay time \( t_z \) is determined from

\[
\text{erf} \left( \sqrt{\frac{t_z}{\tau_{p0}}} \right) + \frac{\exp(-t_z/\tau_{p0})}{\sqrt{\pi t_z/\tau_{p0}}} = 1 + 0.1 \left( \frac{I_f}{I_R} \right)
\]

(8.111)

The total turn-off time is the sum of \( t_s \) and \( t_z \).

To switch the diode quickly, we need to be able to produce a large reverse current as well as have a small minority carrier lifetime. In the design of diode circuits, then, the designer must provide a path for the transient reverse-biased current pulse in order to be able to switch the diode quickly. These same effects will be considered when we discuss the switching of bipolar transistors.

### 8.4.2 The Turn-on Transient

The turn-on transient occurs when the diode is switched from its "off" state into the forward-bias "on" state. The turn-on can be accomplished by applying a forward-bias current pulse. The first stage of turn-on occurs very quickly and is the length of time required to narrow the space charge width from the reverse-biased value to its thermal-equilibrium value when \( V_a = 0 \). During this time, ionized donors and acceptors are neutralized as the space charge width narrows.

The second stage of the turn-on process is the time required to establish the minority carrier distributions. During this time the voltage across the junction is increasing toward its steady-state value. A small turn-on time is achieved if the minority carrier lifetime is small and if the forward-bias current is small.

----

### TEST YOUR UNDERSTANDING

**TYU 8.7** A one-sided p\(^+\)n silicon diode, which has a forward-bias current of \( I_f = 1.75 \, \text{mA} \), is switched to reverse bias with an effective reverse-biased voltage of \( V_R = 2 \, \text{V} \) and an effective series resistance of \( R_s = 4 \, \text{k}\Omega \). The minority carrier hole lifetime is \( 10^{-7} \, \text{s} \). (a) Determine the storage time \( t_s \). (b) Calculate the decay time \( t_z \). (c) What is the turn-off time of the diode?
```