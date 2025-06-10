```markdown
!Figure 11.3

**Figure 11.3** (a) Cross section along channel length of n-channel MOSFET. Energy-band diagrams along channel length at (b) accumulation, (c) weak inversion, and (d) inversion.

Characteristics of a lightly doped n-type material. We would expect, then, to observe some conduction between the n\(^+\) source and drain contacts through this weakly inverted channel. The condition for \(\phi_p < \phi_s < 2\phi_p\) is known as **weak inversion**.

Figure 11.3 shows the surface potential along the length of the channel at accumulation, weak inversion, and threshold for the case when a small drain voltage is applied. The bulk p-substrate is assumed to be at zero potential. Figure 11.3b, c shows the accumulation and weak inversion cases. There is a potential barrier between the n\(^+\) source and channel region which the electrons must overcome in order to generate a channel current. A comparison of these barriers with those in pn junctions would suggest that the channel current is an exponential function of \(V_{GS}\). In the inversion mode, shown in Figure 11.3d, the barrier is so small that we lose the exponential dependence, since the junction is more like an ohmic contact.

The actual derivation of the subthreshold current is beyond the scope of this chapter. We can write that

\[
I_D(\text{sub}) \propto \left[ \exp\left(\frac{eV_{GS}}{kT}\right) \right] \cdot \left[ 1 - \exp\left(\frac{-eV_{DS}}{kT}\right) \right]
\]

(11.1)

If \(V_{DS}\) is larger than a few \((kT/e)\) volts, then the subthreshold current is independent of \(V_{DS}\).

Figure 11.4 shows the exponential behavior of the subthreshold current for several body-to-source voltages. Also shown on the curves are the threshold voltage values. Ideally, a change in gate voltage on approximately 60 mV produces an order of magnitude change in the subthreshold current. A detailed analysis of the subthreshold condition shows that the slope of the \(I_D\) versus \(V_{GS}\) curve is a function of the semiconductor doping and is also a function of the interface state density. The
```