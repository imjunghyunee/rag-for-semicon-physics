## 12.2 Minority Carrier Distribution

This excess concentration will also vary approximately linearly with distance if \( x_E \) is small. We find

\[
\delta p_E(x') \approx \frac{p_{E0}}{x_E} \left[ \exp\left(\frac{eV_{BE}}{kT}\right) - 1 \right] (x_E - x')
\]

(12.21b)

If \( x_E \) is comparable to \( L_E \), then \( \delta p_E(x') \) shows an exponential dependence on \( x_E \).

----

### TEST YOUR UNDERSTANDING

**TYU 12.2** Consider a silicon npn bipolar transistor with emitter and base regions uniformly doped at concentrations of \( 10^{18} \text{cm}^{-3} \) and \( 10^{16} \text{cm}^{-3} \), respectively. A forward-bias B–E voltage of \( V_{BE} = 0.610 \, \text{V} \) is applied. The neutral emitter width is \( x_E = 4 \, \mu \text{m} \) and the minority carrier diffusion length in the emitter is \( L_E = 4 \, \mu \text{m} \). Calculate the excess minority carrier concentration in the emitter at (a) \( x' = 0 \) and (b) \( x' = x_E/2 \).

----

### Collector Region

The excess minority carrier hole concentration in the collector can be determined from the equation

\[
D_C \frac{\partial^2 \delta p_C(x')}{\partial x'^2} - \frac{\delta p_C(x')}{\tau_{C0}} = 0
\]

(12.22)

where \( D_C \) and \( \tau_{C0} \) are the minority carrier diffusion coefficient and minority carrier lifetime, respectively, in the collector. We can express the excess minority carrier hole concentration in the collector as

\[
\delta p_C(x') = p_C(x') - p_{C0}
\]

(12.23)

The general solution to Equation (12.22) can be written as

\[
\delta p_C(x') = G \exp\left(\frac{x'}{L_C}\right) + H \exp\left(-\frac{x'}{L_C}\right)
\]

(12.24)

where \( L_C = \sqrt{D_C \tau_{C0}} \). If we assume that the collector is long, then the coefficient \( G \) must be zero since the excess concentration must remain finite. The second boundary condition gives

\[
\delta p_C(x' = 0) = \delta p_C(0) = p_C(x' = 0) - p_{C0} = 0 - p_{C0} = -p_{C0}
\]

(12.25)

The excess minority carrier hole concentration in the collector is then given as

\[
\delta p_C(x') = -p_{C0} \exp\left(-\frac{x'}{L_C}\right)
\]

(12.26)

This result is exactly what we expect from the results of a reverse-biased pn junction.

----

### TEST YOUR UNDERSTANDING

**TYU 12.3** Consider the collector region of an npn bipolar transistor biased in the forward-active region. At what value of \( x' \), compared to \( L_C \), does the magnitude of the minority carrier concentration reach 95 percent of the thermal-equilibrium value? (\( x' \approx \eta L_C \, \text{where} \, \eta \approx 3 \))