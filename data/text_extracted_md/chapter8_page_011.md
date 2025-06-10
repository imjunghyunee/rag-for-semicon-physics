```markdown
Combining Equations (8.16) and (8.17), we can write

\[
np = n_i^2 \exp \left( \frac{E_{Fn} - E_{Fp}}{kT} \right)
\]

(8.19)

Comparing Equations (8.18) and (8.19), the difference in quasi-Fermi levels is related to the applied bias \( V_a \) and represents the deviation from thermal equilibrium. The difference between \( E_{Fn} \) and \( E_{Fp} \) is nearly constant through the depletion region.

To review, a forward-bias voltage lowers the built-in potential barrier of a pn junction so that electrons from the n region are injected across the space charge region, creating excess minority carriers in the p region. These excess electrons begin diffusing into the bulk p region where they can recombine with majority carrier holes. The excess minority carrier electron concentration then decreases with distance from the junction. The same discussion applies to holes injected across the space charge region into the n region.

### 8.1.5 Ideal pn Junction Current

The approach we use to determine the current in a pn junction is based on the three parts of the fourth assumption stated earlier in this section. The total current in the junction is the sum of the individual electron and hole currents that are constant through the depletion region. Since the electron and hole currents are continuous functions through the pn junction, the total pn junction current will be the minority carrier hole diffusion current at \( x = x_n \) plus the minority carrier electron diffusion current at \( x = -x_p \). The gradients in the minority carrier concentrations, as shown in Figure 8.5, produce diffusion currents, and since we are assuming the electric field to be zero at the space charge edges, we can neglect any minority carrier drift current component. This approach in determining the pn junction current is shown in Figure 8.7.

!Figure 8.7

**Figure 8.7** | Electron and hole current densities through the space charge region of a pn junction.

- \( J_{\text{Total}} = J_n(x_n) + J_p(-x_p) \)
- \( J_n(x_n) \)
- \( J_p(-x_p) \)
```