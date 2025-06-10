```markdown
The collector depletion region transit time is

\[
\tau_d = \frac{x_{dc}}{v_d} = \frac{2.4 \times 10^{-4}}{10^7} = 24 \, \text{ps}
\]

The collector capacitance charging time is

\[
\tau_c = r_c (C_c + C) = (20)(0.2 \times 10^{-12}) = 4 \, \text{ps}
\]

The total emitter-to-collector time delay is then

\[
\tau_{ec} = 25.9 + 50 + 24 + 4 = 103.9 \, \text{ps}
\]

so that the cutoff frequency is calculated as

\[
f_T = \frac{1}{2\pi \tau_{ec}} = \frac{1}{2\pi (103.9 \times 10^{-12})} = 1.53 \, \text{GHz}
\]

If we assume a low-frequency common-emitter current gain of \(\beta = 100\), then the beta cutoff frequency is

\[
f_{\beta} = \frac{f_T}{\beta_0} = \frac{1.53 \times 10^9}{100} = 15.3 \, \text{MHz}
\]

----

**Comment**

The design of high-frequency transistors requires small device geometries in order to reduce capacitances, and narrow base widths in order to reduce the base transit time.

----

**EXERCISE PROBLEM**

**Ex 12.14**  
Consider a bipolar transistor with the same parameters as described in Example 12.14 except that \(I_E = 50 \, \mu A\), \(C_c = 0.40 \, \text{pF}\), and \(C_g = 0.05 \, \text{pF}\). Determine the emitter-to-collector transit time, the cutoff frequency, and the beta cutoff frequency.

\[
(2\text{H} \, \text{V} \, 9\text{S} = 4 \, \text{H} \, \text{V} \, 9\text{S} = 4 \, \text{S} \, \text{d} \, \text{t} \, 8\text{Z} = 2\text{L} \, \text{S} \, \text{V})
\]

----

## 12.7 LARGE-SIGNAL SWITCHING

Switching a transistor from one state to another is strongly related to the frequency characteristics just discussed. However, switching is considered to be a large-signal change, whereas the frequency effects assumed only small changes in the magnitude of the signal.

### 12.7.1 Switching Characteristics

Consider an npn transistor in the circuit shown in Figure 12.43a switching from cutoff to saturation, and then switching back from saturation to cutoff. We describe the physical processes taking place in the transistor during the switching cycle.

Consider, initially, the case of switching from cutoff to saturation. Assume that in cutoff \(V_{BE} = V_{BB} < 0\), thus the B–E junction is reverse biased. At \(t = 0\), assume that \(V_{BB}\) switches to a value of \(V_{BB0}\) as shown in Figure 12.43b. We assume that \(V_{BB0}\) is sufficiently positive to eventually drive the transistor into saturation. For \(0 \leq t \leq t_1\), the base current supplies charge to bring the B–E junction from reverse bias to a slight forward bias. The space charge width of the B–E junction is narrowing, and ionized donors and acceptors are being neutralized. A small amount of charge is also...
```