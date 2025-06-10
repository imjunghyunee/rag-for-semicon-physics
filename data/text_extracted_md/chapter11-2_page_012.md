```markdown
# CHAPTER 11 Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

The threshold voltage after implant, from Equation (11.43), is

\[
V_T = V_{T0} + \frac{eD_I}{C_{ox}}
\]

so that

\[
+0.40 = -0.419 + \frac{(1.6 \times 10^{-19})D_I}{1.9175 \times 10^{-7}}
\]

which gives

\[
D_I = 9.815 \times 10^{11} \text{ cm}^{-2}
\]

If the uniform step implant extends to a depth of \( x_j = 0.15 \, \mu \text{m} \), for example, then the equivalent acceptor concentration at the surface is

\[
N_s - N_a = \frac{D_I}{x_j}
\]

or

\[
N_s - 5 \times 10^{15} = \frac{9.815 \times 10^{11}}{0.15 \times 10^{-4}}
\]

so that

\[
N_s = 7.04 \times 10^{16} \text{ cm}^{-3}
\]

## Comment

It is assumed in the above calculation that the induced space charge width in the channel region is greater than the ion implant depth \( x_j \). The calculation satisfies our assumptions.

## EXERCISE PROBLEM

**Ex 11.6** A silicon MOSFET has the following parameters: \( N_a = 10^{15} \text{ cm}^{-3} \), p\(^+\) polysilicon gate with an initial flat-band voltage of \( V_{FB0} = +0.95 \, \text{V} \), and an oxide thickness of \( t_{ox} = 12 \, \text{nm} = 120 \, \text{Å} \). A final threshold voltage of \( V_T = +0.40 \, \text{V} \) is required. Assume the idealized delta function for the ion implant profile shown in Figure 11.29a. (a) What type of ion (acceptor or donor) should be implanted? (b) Determine the required ion dose \( D_I \).

The actual implant dose versus distance is neither a delta function nor a step function; it tends to be a Gaussian-type distribution. The threshold shift due to a nonuniform ion implant density may be defined as the shift in curves of \( N_{inv} \) versus \( V_G \), where \( N_{inv} \) is the inversion carrier density per cm\(^2\). This shift corresponds to an experimental shift of drain current versus \( V_G \) when the transistor is biased in the linear mode. The criteria of the threshold inversion point as \( \phi_s = 2\phi_f \) in the implanted devices have an uncertain meaning because of the nonuniform doping in the substrate. The determination of the threshold voltage becomes more complicated and will not be made here.

## TEST YOUR UNDERSTANDING

**TYU 11.4** Repeat Exercise Problem Ex 11.6 for a final threshold voltage of  
(a) \( V_T = +0.25 \, \text{V} \) and (b) \( V_T = -0.25 \, \text{V} \).
```