```markdown
# CHAPTER 11 Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

!Figure 11.13  
**Figure 11.13** | Cross section of a long n-channel MOSFET (a) at flat band and (b) at inversion.

The semiconductor space charge region is zero. We assumed that the gate area was the same as the active area in the semiconductor. Using this assumption, we have considered only equivalent surface charge densities and neglected any effects on threshold voltage that may occur due to source and drain space charge regions that extend into the active channel region.

Figure 11.13a shows the cross section of a long n-channel MOSFET at flat band, with zero source and drain voltage applied. The space charge regions at the source and drain extend into the channel region but occupy only a small fraction of the entire channel region. The gate voltage, then, will control essentially all of the space charge induced in the channel region at inversion as shown in Figure 11.13b.

As the channel length decreases, the fraction of charge in the channel region controlled by the gate decreases. This effect can be seen in Figure 11.14 for the flat-band condition. As the drain voltage increases, the reverse-biased space charge region at the drain extends further into the channel area and the gate controls even less bulk charge. The amount of charge in the channel region, \( Q_{SD}(max) \), controlled by the gate, affects the threshold voltage and can be seen from Equation (11.24)

\[
V_{TN} = (|Q_{SD}(max)| - Q'_{I}) \left( \frac{t_{ox}}{\varepsilon_{ox}} \right) + \phi_{ms} + 2\phi_{fp}
\]

(11.24)

We can quantitatively determine the short-channel effects on the threshold voltage by considering the parameters shown in Figure 11.15. The source and drain junctions are characterized by a diffused junction depth \( x_j \). We will assume that the lateral diffusion distance under the gate is the same as the vertical diffusion distance. This assumption is a reasonably good approximation for diffused junctions but becomes less accurate for ion implanted junctions. We will initially consider the case when the source, drain, and body contacts are all at ground potential.

The basic assumption in this analysis is that the bulk charge in the trapezoidal region under the gate is controlled by the gate. The potential difference across the bulk space charge region is \( 2\phi_{fp} \) at the threshold inversion point, and the built-in potential barrier height of the source and drain junctions is also approximately \( 2\phi_{fp} \), implying that the three space charge widths are essentially equal. We can then write

\[
x_s \approx x_d \approx x_{eff} = x_{jT}
\]

(11.25)
```