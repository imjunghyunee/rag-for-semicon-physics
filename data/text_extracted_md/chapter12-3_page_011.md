```markdown
The integral in the denominator is the total majority carrier charge in the base and is known as the **base Gummel number**, defined as \( Q_B \).

If we perform the same analysis in the emitter, we find that the hole current density in the emitter of an npn transistor can be expressed as

\[
J_p = -eD_p \frac{n_i^2}{N_D} \exp(V_{BE}/V_t) \left/ \int_0^{x_n} n_{E}^{'c}(x') \, dx' \right.
\]

(12.85)

The integral in the denominator is the total majority carrier charge in the emitter and is known as the **emitter Gummel number**, defined as \( Q_E \).

Since the currents in the Gummel–Poon model are functions of the total integrated charges in the base and emitter, these currents can easily be determined for nonuniformly doped transistors.

The Gummel–Poon model can also take into account nonideal effects, such as the Early effect and high-level injection. As the B–C voltage changes, the neutral base width changes so that the base Gummel number \( Q_B \) changes. The change in \( Q_B \) with B–C voltage then makes the electron current density given by Equation (12.84) a function of the B–C voltage. This is the base width modulation effect or Early effect as discussed previously in Section 12.4.1.

If the B–E voltage becomes too large, low injection no longer applies, which leads to high-level injection. In this case, the total hole concentration in the base increases because of the increased excess hole concentration. This means that the base Gummel number will increase. The change in base Gummel number implies, from Equation (12.84), that the electron current density will also change. High-level injection has also been previously discussed in Section 12.4.2.

The Gummel–Poon model can then be used to describe the basic operation of the transistor as well as to describe nonideal effects.

## 12.5.3 Hybrid-Pi Model

Bipolar transistors are commonly used in circuits that amplify time-varying or sinusoidal signals. In these linear amplifier circuits, the transistor is biased in the forward-active region and small sinusoidal voltages and currents are superimposed on dc voltages and currents. In these applications, the sinusoidal parameters are of interest, so it is convenient to develop a small-signal equivalent circuit of the bipolar transistor using the small-signal admittance parameters of the pn junction developed in Chapter 8.

Figure 12.38a shows an npn bipolar transistor in a common-emitter configuration with the small-signal terminal voltages and currents. Figure 12.38b shows the cross section of the npn transistor. The C, B, and E terminals are the external connections to the transistor, while the \( C' \), \( B' \), and \( E' \) points are the idealized internal collector, base, and emitter regions.

We can begin constructing the equivalent circuit of the transistor by considering the various terminals individually. Figure 12.39a shows the equivalent circuit between the external input base terminal and the external emitter terminal. The resistance \( r_p \) is the series resistance in the base between the external base terminal B and the internal base region \( B' \). The \( B'–E' \) junction is forward biased, so \( C_{\pi} \) is
```