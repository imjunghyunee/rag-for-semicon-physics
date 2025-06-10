```markdown
# 12.4 Nonideal Effects

## Objective
Design the collector doping concentration and collector width to meet a punch-through voltage specification.

Consider a uniformly doped silicon bipolar transistor with a metallurgical base width of 0.5 μm and a base doping of \( N_B = 10^{16} \, \text{cm}^{-3} \). The punch-through voltage is to be \( V_{PT} = 25 \, \text{V} \).

## Solution
The maximum collector doping concentration can be determined from Equation (12.54) as:

\[
25 = \frac{(1.6 \times 10^{-19})(0.5 \times 10^{-4})(10^{16})(N_C + 10^{16})}{2(11.7)(8.85 \times 10^{-14})N_C}
\]

or

\[
12.94 = 1 + \frac{10^{16}}{N_C}
\]

which yields

\[
N_C = 8.38 \times 10^{14} \, \text{cm}^{-3}
\]

Using this n-type doping concentration for the collector, we can determine the minimum width of the collector region such that the depletion region extending into the collector will not reach the substrate and cause breakdown in the collector region. We have, using the results of Chapter 7,

\[
x_{sc} = x_c = 5.97 \, \mu\text{m}
\]

## Comment
From Figure 7.15, the expected avalanche breakdown voltage for this junction is greater than 300 V. Obviously punch-through will occur before the normal breakdown voltage in this case. For a larger punch-through voltage, a larger metallurgical base width will be required, since a lower collector doping concentration is becoming impractical. A larger punch-through voltage will also require a larger collector width in order to avoid premature breakdown in this region.

## Exercise Problem

**Ex 12.10** The metallurgical base width of a silicon npn bipolar transistor is \( x_{BO} = 0.80 \, \mu\text{m} \).

The base and collector doping concentrations are \( N_B = 5 \times 10^{16} \, \text{cm}^{-3} \) and \( N_C = 2 \times 10^{15} \, \text{cm}^{-3} \), respectively. 

(a) Determine the punch-through voltage.  
(b) What is the expected avalanche breakdown voltage?  
[Answer: (a) 81 V; (b) 519 V]

The second breakdown mechanism to consider is avalanche breakdown, but taking into account the gain of the transistor. Figure 12.34a is an npn transistor with a reverse-biased voltage applied to the B–C junction and with the emitter left open. The current \( I_{BCO} \) is the reverse-biased junction current. Figure 12.34b shows the transistor with an applied C–E voltage and with the base terminal left open. This bias

*The doping concentrations in the base and collector of the transistor are small enough that Zener breakdown is not a factor to be considered.*
```