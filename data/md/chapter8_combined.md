# Chapter 8: The pn Junction Diode

In the last chapter, we discussed the electrostatics of the pn junction in thermal equilibrium and under reverse bias. We determined the built-in potential barrier at thermal equilibrium and calculated the electric field in the space charge region. We also considered the junction capacitance.

In this chapter, we consider the pn junction with a forward-bias voltage applied and determine the current–voltage characteristics. The potential barrier of the pn junction is lowered when a forward-bias voltage is applied, allowing electrons and holes to flow across the space charge region. When holes flow from the p region across the space charge region into the n region, they become excess minority carrier holes and are subject to the excess minority carrier diffusion, drift, and recombination processes discussed in Chapter 6. Likewise, when electrons from the n region flow across the space charge region into the p region, they become excess minority carrier electrons and are subject to these same processes.

## 8.0 | Preview

In this chapter, we will:

- Consider the process by which the potential barrier of a pn junction is lowered when a forward-bias voltage is applied, so holes and electrons can flow across the junction generating a diode current.
- Derive the boundary conditions for excess holes in the n region and excess electrons in the p region, and analyze the behavior of these excess carriers under a forward bias.
- Derive the ideal current–voltage relation of the forward-biased pn junction diode.
- Describe and analyze nonideal effects in the pn junction diode such as high-level injection, and generation and recombination currents.
- Develop a small-signal equivalent circuit of the pn junction diode. This equivalent circuit is used to relate small time-varying currents and voltages in the pn junction.

NO_CONTENT_HERE

```markdown
# CHAPTER 8 The pn Junction Diode

in the n region is lower than that in the p region. The total potential barrier is now larger than that for the zero-bias case. We argued in the last chapter that the increased potential barrier continues to hold back the electrons and holes so that there is still essentially no charge flow and hence essentially no current.

Figure 8.1c shows the energy-band diagram for the case when a positive voltage is applied to the p region with respect to the n region. The Fermi level in the p region is now lower than that in the n region. The total potential barrier is now reduced. The smaller potential barrier means that the electric field in the depletion region is also reduced. The smaller electric field means that the electrons and holes are no longer held back in the n and p regions, respectively. There will be a diffusion of holes from the p region across the space charge region where they will flow into the n region. Similarly, there will be a diffusion of electrons from the n region across the space charge region where they will flow into the p region. The flow of charge generates a current through the pn junction.

The injection of holes into the n region means that these holes are minority carriers. Likewise, the injection of electrons into the p region means that these electrons are minority carriers. The behavior of these minority carriers is described by the ambipolar transport equations that were discussed in Chapter 6. There will be diffusion as well as recombination of excess carriers in these regions. The diffusion of carriers implies that there will be diffusion currents. The mathematical derivation of the pn junction current–voltage relationship is considered in the next section.

## 8.1.2 Ideal Current–Voltage Relationship

The ideal current–voltage relationship of a pn junction is derived on the basis of four assumptions. (The last assumption has three parts, but each part deals with current.) They are:

1. **The abrupt depletion layer approximation applies.** The space charge regions have abrupt boundaries, and the semiconductor is neutral outside of the depletion region.

2. **The Maxwell–Boltzmann approximation applies to carrier statistics.**

3. **The concepts of low injection and complete ionization apply.**

4. **The total current is a constant throughout the entire pn structure.**

   4a. The total current is a constant throughout the entire pn structure.

   4b. The individual electron and hole currents are continuous functions through the pn structure.

   4c. The individual electron and hole currents are constant throughout the depletion region.

Notation can sometimes appear to be overwhelming in the equations in this chapter. Table 8.1 lists some of the various electron and hole concentration terms that appear. Many terms have already been used in previous chapters but are repeated here for convenience.
```

```markdown
## Table 8.1.1 Commonly used terms and notation for this chapter

| Term       | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| \( N_a \)  | Acceptor concentration in the p region of the pn junction               |
| \( N_d \)  | Donor concentration in the n region of the pn junction                  |
| \( n_{eo} = N_d \) | Thermal-equilibrium majority carrier electron concentration in the n region |
| \( p_{eo} = N_a \) | Thermal-equilibrium majority carrier hole concentration in the p region |
| \( n_{po} = \frac{n_i^2}{N_a} \) | Thermal-equilibrium minority carrier electron concentration in the p region |
| \( p_{no} = \frac{n_i^2}{N_d} \) | Thermal-equilibrium minority carrier hole concentration in the n region |
| \( n_p \)  | Total minority carrier electron concentration in the p region           |
| \( p_n \)  | Total minority carrier hole concentration in the n region               |
| \( n_p(x) \) | Minority carrier electron concentration in the p region at the space charge edge |
| \( p_n(x) \) | Minority carrier hole concentration in the n region at the space charge edge |
| \( \delta n_p = n_p - n_{po} \) | Excess minority carrier electron concentration in the p region |
| \( \delta p_n = p_n - p_{no} \) | Excess minority carrier hole concentration in the n region |

### 8.1.3 Boundary Conditions

Figure 8.2 shows the conduction-band energy through the pn junction in thermal equilibrium. The n region contains many more electrons in the conduction band than the p region; the built-in potential barrier prevents this large density of electrons from flowing into the p region. The built-in potential barrier maintains equilibrium between the carrier distributions on either side of the junction.

An expression for the built-in potential barrier was derived in the last chapter and was given by Equation (7.10) as

\[
V_{bi} = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right)
\]

!Figure 8.2  
**Figure 8.2.1** Conduction-band energy through a pn junction.
```

# Chapter 8: The pn Junction Diode

If we divide the equation by \( V_t = kT/e \), take the exponential of both sides, and then take the reciprocal, we obtain

\[
\frac{n_i^2}{N_D N_A} = \exp\left(-\frac{eV_0}{kT}\right) \tag{8.1}
\]

If we assume complete ionization, we can write

\[
n_{n0} \approx N_D \tag{8.2}
\]

where \( n_{n0} \) is the thermal-equilibrium concentration of majority carrier electrons in the n region. In the p region, we can write

\[
n_{p0} \approx \frac{n_i^2}{N_A} \tag{8.3}
\]

where \( n_{p0} \) is the thermal-equilibrium concentration of minority carrier electrons. Substituting Equations (8.2) and (8.3) into Equation (8.1), we obtain

\[
n_{p0} = n_{n0} \exp\left(-\frac{eV_0}{kT}\right) \tag{8.4}
\]

This equation relates the minority carrier electron concentration on the p side of the junction to the majority carrier electron concentration on the n side of the junction in thermal equilibrium.

If a positive voltage is applied to the p region with respect to the n region, the potential barrier is reduced. Figure 8.3a shows a pn junction with an applied voltage \( V_a \). The electric field in the bulk p and n regions is normally very small. Essentially all of the applied voltage is across the junction region. The electric field \( E_{\text{app}} \) induced by the applied voltage is in the opposite direction to the thermal-equilibrium space charge electric field, so the net electric field in the space charge region is reduced below the equilibrium value. The delicate balance between diffusion and the E-field

!Figure 8.3

**Figure 8.3** (a) A pn junction with an applied forward-bias voltage showing the directions of the electric field induced by \( V_a \) and the space charge electric field. (b) Energy-band diagram of the forward-biased pn junction.

### 8.1 pn Junction Current

force achieved at thermal equilibrium is upset. The electric field force that prevented majority carriers from crossing the space charge region is reduced; majority carrier electrons from the n side are now injected across the depletion region into the p material, and majority carrier holes from the p side are injected across the depletion region into the n material. As long as the bias \( V_a \) is applied, the injection of carriers across the space charge region continues and a current is created in the pn junction. This bias condition is known as forward bias; the energy-band diagram of the forward-biased pn junction is shown in Figure 8.3b.

The potential barrier \( V_0 \) in Equation (8.4) can be replaced by \( (V_0 - V_a) \) when the junction is forward biased. Equation (8.4) becomes

\[
n_p = n_{p0} \exp \left( \frac{-e(V_0 - V_a)}{kT} \right) = n_{p0} \exp \left( \frac{-eV_0}{kT} \right) \exp \left( \frac{+eV_a}{kT} \right)
\]

(8.5)

If we assume low injection, the majority carrier electron concentration \( n_{n0} \), for example, does not change significantly. However, the minority carrier concentration, \( n_p \), can deviate from its thermal-equilibrium value \( n_{p0} \) by orders of magnitude. Using Equation (8.4), we can write Equation (8.5) as

\[
n_p = n_{p0} \exp \left( \frac{eV_a}{kT} \right)
\]

(8.6)

When a forward-bias voltage is applied to the pn junction, the junction is no longer in thermal equilibrium. The left side of Equation (8.6) is the total minority carrier electron concentration in the p region, which is now greater than the thermal equilibrium value. The forward-bias voltage lowers the potential barrier so that majority carrier electrons from the n region are injected across the junction into the p region, thereby increasing the minority carrier electron concentration. We have produced excess minority carrier electrons in the p region.

When the electrons are injected into the p region, these excess carriers are subject to the diffusion and recombination processes we discussed in Chapter 6. Equation (8.6), then, is the expression for the minority carrier electron concentration at the edge of the space charge region in the p region.

Exactly the same process occurs for majority carrier holes in the p region, which are injected across the space charge region into the n region under a forward-bias voltage. We can write that

\[
p_n = p_{n0} \exp \left( \frac{eV_a}{kT} \right)
\]

(8.7)

where \( p_n \) is the concentration of minority carrier holes at the edge of the space charge region in the n region. Figure 8.4 shows these results. By applying a forward-bias voltage, we create excess minority carriers in each region of the pn junction.

# Chapter 8: The pn Junction Diode

!Figure 8.4

**Figure 8.4** Excess minority carrier concentrations at the space charge edges generated by the forward-bias voltage.

## Example 8.1

### Objective

Calculate the minority carrier concentrations at the edge of the space charge regions in a forward-biased pn junction.

Consider a silicon pn junction at \( T = 300 \, \text{K} \). Assume the doping concentration in the n region is \( N_D = 10^{16} \, \text{cm}^{-3} \) and the doping concentration in the p region is \( N_A = 6 \times 10^{15} \, \text{cm}^{-3} \), and assume that a forward bias of 0.60 V is applied to the pn junction.

### Solution

From Equations (8.6) and (8.7) and from Figure 8.4, we have

\[
n_p(-x_p) = n_{po} \exp\left(\frac{eV}{kT}\right) \quad \text{and} \quad p_n(x_n) = p_{no} \exp\left(\frac{eV}{kT}\right)
\]

The thermal-equilibrium minority carrier concentrations are

\[
n_{po} = \frac{n_i^2}{N_A} = \frac{(1.5 \times 10^{10})^2}{6 \times 10^{15}} = 3.75 \times 10^4 \, \text{cm}^{-3}
\]

and

\[
p_{no} = \frac{n_i^2}{N_D} = \frac{(1.5 \times 10^{10})^2}{10^{16}} = 2.25 \times 10^4 \, \text{cm}^{-3}
\]

We then have

\[
n_p(-x_p) = 3.75 \times 10^4 \exp\left(\frac{0.60}{0.0259}\right) = 4.31 \times 10^{14} \, \text{cm}^{-3}
\]

and

\[
p_n(x_n) = 2.25 \times 10^4 \exp\left(\frac{0.60}{0.0259}\right) = 2.59 \times 10^{14} \, \text{cm}^{-3}
\]

### Comment

The minority carrier concentrations can increase by many orders of magnitude when a relatively small forward-bias voltage is applied. Low injection still applies, however, since the excess minority carrier concentrations at the space-charge edges are much less than the thermal-equilibrium majority carrier concentrations.

### 8.1 pn Junction Current

#### Exercise Problem

**Ex 8.1**  
A silicon pn junction at \( T = 300 \, \text{K} \) is doped with impurity concentrations of \( N_d = 2 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). The junction is forward biased at \( V_a = 0.650 \, \text{V} \). Determine the minority carrier concentrations at the space charge edges. Does low injection still apply?  
\[ \text{[S^a \cdot t \cdot uw} \, 0.1 \times ZG \, 8 = (\gamma \cdot d \cdot uw \, 0.1) \times LS \cdot \varepsilon = (t-x-y \, t+su \, v] \]

The minority carrier concentrations at the space charge edges, given by Equations (8.6) and (8.7), were derived assuming that a forward-bias voltage (\( V_a > 0 \)) was applied across the pn junction. However, nothing in the derivation prevents \( V_a \) from being negative (reverse bias). If a reverse-biased voltage greater than a few tenths of a volt is applied to the pn junction, then we see from Equations (8.6) and (8.7) that the minority carrier concentrations at the space charge edge are essentially zero. The minority carrier concentrations for the reverse-biased condition drop below the thermal-equilibrium values.

### 8.1.4 Minority Carrier Distribution

We developed, in Chapter 6, the ambipolar transport equation for excess minority carrier holes in an n region. This equation, in one dimension, is

\[
D_p \frac{\partial^2 (\delta p_n)}{\partial x^2} - \mu_p E \frac{\partial (\delta p_n)}{\partial x} + g' - \frac{\delta p_n}{\tau_{p0}} = \frac{\partial (\delta p_n)}{\partial t}
\]

(8.8)

where \( \delta p_n = p_n - p_{n0} \) is the excess minority carrier hole concentration and is the difference between the total and thermal equilibrium minority carrier concentrations. The ambipolar transport equation describes the behavior of excess carriers as a function of time and spatial coordinates.

In Chapter 5, we calculated drift current densities in a semiconductor. We determined that relatively large currents could be created with fairly small electric fields. As a first approximation, we assume that the electric field is zero in both the neutral p and n regions. In the n region for \( x > x_n \), we have that \( E = 0 \) and \( g' = 0 \). If we also assume steady state so \( \partial (\delta p_n)/\partial t = 0 \), then Equation (8.8) reduces to

\[
\frac{d^2 (\delta p_n)}{dx^2} - \frac{\delta p_n}{L_{p}^2} = 0 \quad (x > x_n)
\]

(8.9)

where \( L_{p}^2 = D_p \tau_{p0} \). For the same set of conditions, the excess minority carrier electron concentration in the p region is determined from

\[
\frac{d^2 (\delta n_p)}{dx^2} - \frac{\delta n_p}{L_{n}^2} = 0 \quad (x < x_p)
\]

(8.10)

where \( L_{n}^2 = D_n \tau_{n0} \).

# Chapter 8: The pn Junction Diode

The boundary conditions for the total minority carrier concentrations are

\[
p_n(x_n) = p_{n0} \exp\left(\frac{eV_a}{kT}\right) \tag{8.11a}
\]

\[
n_p(-x_p) = n_{p0} \exp\left(\frac{eV_a}{kT}\right) \tag{8.11b}
\]

\[
p_n(x \to +\infty) = p_{n0} \tag{8.11c}
\]

\[
n_p(x \to -\infty) = n_{p0} \tag{8.11d}
\]

As minority carriers diffuse from the space charge edge into the neutral semiconductor regions, they recombine with majority carriers. We assume that the lengths \(W_n\) and \(W_p\) shown in Figure 8.3a are very long, meaning in particular that \(W_n \gg L_n\) and \(W_p \gg L_p\). The excess minority carrier concentrations must approach zero at distances far from the space charge region. The structure is referred to as a long pn junction.

The general solution to Equation (8.9) is

\[
\delta p_n(x) = p_n(x) - p_{n0} = A e^{+x/L_p} + B e^{-x/L_p} \quad (x \geq x_n) \tag{8.12}
\]

and the general solution to Equation (8.10) is

\[
\delta n_p(x) = n_p(x) - n_{p0} = C e^{+x/L_n} + D e^{-x/L_n} \quad (x \leq -x_p) \tag{8.13}
\]

Applying the boundary conditions from Equations (8.11c) and (8.11d), the coefficients \(A\) and \(D\) must be zero. The coefficients \(B\) and \(C\) may be determined from the boundary conditions given by Equations (8.11a) and (8.11b). The excess carrier concentrations are then found to be, for \(x \geq x_n\),

\[
\delta p_n(x) = p_n(x) - p_{n0} = p_{n0} \left[\exp\left(\frac{eV_a}{kT}\right) - 1\right] \exp\left(\frac{x - x_n}{L_p}\right) \tag{8.14}
\]

and, for \(x \leq -x_p\),

\[
\delta n_p(x) = n_p(x) - n_{p0} = n_{p0} \left[\exp\left(\frac{eV_a}{kT}\right) - 1\right] \exp\left(\frac{x + x_p}{L_n}\right) \tag{8.15}
\]

The minority carrier concentrations decay exponentially with distance away from the junction to their thermal-equilibrium values. Figure 8.5 shows these results. Again, we have assumed that both the n-region and the p-region lengths are long compared to the minority carrier diffusion lengths.

In Chapter 6, we discussed the concept of quasi-Fermi levels, which apply to excess carriers in a nonequilibrium condition. Since excess electrons exist in the neutral p region and excess holes exist in the neutral n region, we can apply quasi-Fermi levels to these regions. We had defined quasi-Fermi levels in terms of carrier concentrations as

\[
p = p_0 + \delta p = n_i \exp\left(\frac{E_{Fi} - E_{Fp}}{kT}\right) \tag{8.16}
\]

### 8.1 pn Junction Current

!Figure 8.5

**Figure 8.5** | Steady-state minority carrier concentrations in a pn junction under forward bias.

!Figure 8.6

**Figure 8.6** | Quasi-Fermi levels through a forward-biased pn junction.

and

\[
n = n_0 + \delta n = n_i \exp\left(\frac{E_{Fn} - E_{Fi}}{kT}\right)
\]

(8.17)

Figure 8.6 shows the quasi-Fermi levels through the pn junction. From Equations (8.14) and (8.15), the carrier concentrations are exponential functions of distance, and from Equations (8.16) and (8.17), the carrier concentrations are exponential functions of the quasi-Fermi levels. The quasi-Fermi levels are then linear functions of distance in the neutral p and n regions as shown in Figure 8.6.

We may note that close to the space charge edge in the p region, \(E_{Fn} - E_{Fi} > 0\) which means that \(\delta n > n_i\). Further from the space charge edge, \(E_{Fn} - E_{Fi} < 0\) which means that \(\delta n < n_i\) and the excess electron concentration is approaching zero. The same discussion applies to the excess hole concentration in the n region.

At the space charge edge at \(x = x_n\), we can write, for low injection

\[
n_e p_e (x_n) = n_e p_{e0} \exp\left(\frac{V_a}{V_t}\right) = n_i^2 \exp\left(\frac{V_a}{V_t}\right)
\]

(8.18)

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

# 8.1 pn Junction Current

We can calculate the minority carrier hole diffusion current density at \( x = x_n \) from the relation

\[
J_p(x_n) = -eD_p \left. \frac{dp(x)}{dx} \right|_{x=x_n}
\]

(8.20)

Since we are assuming uniformly doped regions, the thermal-equilibrium carrier concentration is constant, so the hole diffusion current density may be written as

\[
J_p(x_n) = -eD_p \left. \frac{d\Delta p(x)}{dx} \right|_{x=x_n}
\]

(8.21)

Taking the derivative of Equation (8.14) and substituting into Equation (8.21), we obtain

\[
J_p(x_n) = \frac{eD_p n_{p0}}{L_p} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.22)

The hole current density for this forward-bias condition is in the \( +x \) direction, which is from the p to the n region.

Similarly, we may calculate the electron diffusion current density at \( x = -x_p \). This may be written as

\[
J_n(-x_p) = eD_n \left. \frac{d\Delta n(x)}{dx} \right|_{x=-x_p}
\]

(8.23)

Using Equation (8.15), we obtain

\[
J_n(-x_p) = \frac{eD_n n_{p0}}{L_n} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.24)

The electron current density is also in the \( +x \) direction.

An assumption we made at the beginning was that the individual electron and hole currents were continuous functions and constant through the space charge region. The total current is the sum of the electron and hole currents and is constant through the entire junction. Figure 8.7 again shows a plot of the magnitudes of these currents.

The total current density in the pn junction is then

\[
J = J_p(x_n) + J_n(-x_p) = \left[ \frac{eD_p n_{p0}}{L_p} + \frac{eD_n n_{p0}}{L_n} \right] \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.25)

Equation (8.25) is the ideal current–voltage relationship of a pn junction.

We may define a parameter \( J_s \) as

\[
J_s = \frac{eD_p n_{p0}}{L_p} + \frac{eD_n n_{p0}}{L_n}
\]

(8.26)

# CHAPTER 8 The pn Junction Diode

!Figure 8.8

**Figure 8.8** Ideal \( I \text{-} V \) characteristic of a pn junction diode.

so that Equation (8.25) may be written as

\[
J = J_s \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.27)

Equation (8.27), known as the ideal-diode equation, gives a good description of the current–voltage characteristics of the pn junction over a wide range of currents and voltages. Although Equation (8.27) was derived assuming a forward-bias voltage (\( V_a > 0 \)), there is nothing to prevent \( V_a \) from being negative (reverse bias). Equation (8.27) is plotted in Figure 8.8 as a function of forward-bias voltage \( V_a \). If the voltage \( V_a \) becomes negative (reverse bias) by a few kT/eV, then the reverse-biased current density becomes independent of the reverse-biased voltage. The parameter \( J_s \) is then referred to as the reverse-saturation current density. The current–voltage characteristics of the pn junction diode are obviously not bilateral.

----

**EXAMPLE 8.2**

**Objective:** Determine the ideal reverse-saturation current density in a silicon pn junction at \( T = 300 \, \text{K} \).

Consider the following parameters in a silicon pn junction:

\[
\begin{align*}
N_d = N_a &= 10^{16} \, \text{cm}^{-3} \\
D_n &= 25 \, \text{cm}^2/\text{s} \\
D_p &= 10 \, \text{cm}^2/\text{s} \\
n_i &= 1.5 \times 10^{10} \, \text{cm}^{-3} \\
\tau_p = \tau_n &= 5 \times 10^{-7} \, \text{s} \\
\epsilon &= 11.7
\end{align*}
\]

**Solution**

The ideal reverse-saturation current density is given by

\[
J_s = eD_n \frac{n_{i0}}{L_n} + eD_p \frac{p_{i0}}{L_p}
\]

## 8.1 pn Junction Current

which may be rewritten as

\[
J_s = en_i^2 \left( \frac{1}{N_A} \sqrt{\frac{D_n}{\tau_{n0}}} + \frac{1}{N_D} \sqrt{\frac{D_p}{\tau_{p0}}} \right)
\]

Then

\[
J_s = (1.6 \times 10^{-19})(1.5 \times 10^{10})^2 \left( \frac{1}{10^{16}} \sqrt{\frac{25}{5 \times 10^{-7}}} + \frac{1}{10^{16}} \sqrt{\frac{10}{5 \times 10^{-7}}} \right)
\]

or

\[
J_s = 4.16 \times 10^{-11} \, \text{A/cm}^2
\]

### Comment

The ideal reverse-biased saturation current density is very small. If the pn junction cross-sectional area were \( A = 10^{-4} \, \text{cm}^2 \), for example, then the ideal reverse-biased diode current would be \( I_s = 4.15 \times 10^{-15} \, \text{A} \).

### EXERCISE PROBLEM

**Ex 8.2** Consider a GaAs pn junction diode at \( T = 300 \, \text{K} \). The parameters of the device are \( N_D = 2 \times 10^{16} \, \text{cm}^{-3}, N_A = 8 \times 10^{15} \, \text{cm}^{-3}, D_n = 210 \, \text{cm}^2/\text{s}, D_p = 8 \, \text{cm}^2/\text{s}, \tau_{n0} = 10^{-7} \, \text{s}, \) and \( \tau_{p0} = 5 \times 10^{-8} \, \text{s} \). Determine the ideal reverse-saturation current density.

\((\mu \text{m}/y = 0.1 \, \text{X} \, 0^{\circ} \text{C} \, \text{E} = f' \, \text{SUY})\)

If the forward-bias voltage in Equation (8.27) is positive by more than a few \( kT/eV \), then the \((-1)\) term in Equation (8.27) becomes negligible. Figure 8.9 shows the forward-bias current–voltage characteristic when the current is plotted on a log scale. Ideally, this plot yields a straight line when \( V_a \) is greater than a few \( kT/eV \). The forward-bias current is an exponential function of the forward-bias voltage.

!Figure 8.9

**Figure 8.9** | Ideal \( I-V \) characteristic of a pn junction diode with the current plotted on a log scale.

```markdown
# CHAPTER 8 The pn Junction Diode

## EXAMPLE 8.3

**Objective:** Design a pn junction diode to produce particular electron and hole current densities at a given forward-bias voltage.

Consider a silicon pn junction diode at \( T = 300 \, \text{K} \). Design the diode such that \( J_n = 20 \, \text{A/cm}^2 \) and \( J_p = 5 \, \text{A/cm}^2 \) at \( V_a = 0.65 \, \text{V} \). Assume the remaining semiconductor parameters are as given in Example 8.2.

### Solution

The electron diffusion current density is given by Equation (8.24) as

\[
J_n = e D_n \frac{n_{po}}{L_n} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] = e \sqrt{\frac{D_n}{\tau_{po}}} \frac{n_i^2}{N_a} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

Substituting the numbers, we have

\[
20 = (1.6 \times 10^{-19}) \sqrt{\frac{25}{5 \times 10^{-7}}} \frac{(1.5 \times 10^{10})^2}{N_a} \left[ \exp \left( \frac{0.65}{0.0259} \right) - 1 \right]
\]

which yields

\[
N_a = 1.01 \times 10^{15} \, \text{cm}^{-3}
\]

The hole diffusion current density is given by Equation (8.22) as

\[
J_p = e D_p \frac{p_{no}}{L_p} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] = e \sqrt{\frac{D_p}{\tau_{no}}} \frac{n_i^2}{N_d} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

Substituting the numbers, we have

\[
5 = (1.6 \times 10^{-19}) \sqrt{\frac{10}{5 \times 10^{-7}}} \frac{(1.5 \times 10^{10})^2}{N_d} \left[ \exp \left( \frac{0.65}{0.0259} \right) - 1 \right]
\]

which yields

\[
N_d = 2.55 \times 10^{15} \, \text{cm}^{-3}
\]

### Comment

The relative magnitude of the electron and hole current densities through a diode can be varied by changing the doping concentrations in the device.

### EXERCISE PROBLEM

**Ex 8.3** Using the parameters given in Ex 8.2 for the GaAs diode, determine the electron and hole current densities at the space charge edges, and determine the total current density in the diode for a forward-bias voltage of \( V_a = 1.05 \, \text{V} \).

## 8.1.6 Summary of Physics

We have been considering the case of a forward-bias voltage being applied to a pn junction. The forward-bias voltage lowers the potential barrier so that electrons and holes are injected across the space charge region. The injected carriers become minority carriers which then diffuse from the junction and recombine with majority carriers.

We calculated the minority carrier diffusion current densities at the edge of the space charge region. We can reconsider Equations (8.14) and (8.15) and determine...
```

!Figure 8.10

**Figure 8.10** Ideal electron and hole current components through a pn junction under forward bias.

The minority carrier diffusion current densities as a function of distance through the p and n regions. These results are

\[
J_{p}(x) = \frac{eD_{p}n_{0}}{L_{p}} \left[ \exp\left(\frac{eV_{a}}{kT}\right) - 1 \right] \exp\left(\frac{x_{n} - x}{L_{p}}\right) \quad (x \geq x_{n})
\]

(8.28)

and

\[
J_{n}(x) = \frac{eD_{n}p_{0}}{L_{n}} \left[ \exp\left(\frac{eV_{a}}{kT}\right) - 1 \right] \exp\left(\frac{x_{p} + x}{L_{n}}\right) \quad (x \leq -x_{p})
\]

(8.29)

The minority carrier diffusion current densities decay exponentially in each region. However, the total current through the pn junction is constant. The difference between total current and minority carrier diffusion current is a majority carrier current. Figure 8.10 shows the various current components through the pn structure. The drift of majority carrier holes in the p region far from the junction, for example, is to supply holes that are being injected across the space charge region into the n region and also to supply holes that are lost by recombination with excess minority carrier electrons. The same discussion applies to the drift of electrons in the n region.

We have seen that excess carriers are created in a forward-biased pn junction. From the results of the ambipolar transport theory derived in Chapter 6, the behavior of the excess carriers is determined by the minority carrier parameters for low injection. In determining the current–voltage relationship of the pn junction, we consider the flow of minority carriers since we know the behavior and characteristics of these particles. It may seem strange, at times, that we concern ourselves so much with minority carriers rather than with the vast number of majority carriers, but the reason for this can be found in the results derived from the ambipolar transport theory.

# CHAPTER 8 The pn Junction Diode

The fact that we now have drift current densities in the p and n regions implies that the electric field in these regions is not zero as we had originally assumed. We can calculate the electric field in the neutral regions and determine the validity of our zero-field approximation.

## EXAMPLE 8.4

**Objective:** Calculate the electric field in a neutral region of a silicon diode to produce a given majority carrier drift current density.

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with the parameters given in Example 8.2 and with an applied forward-bias voltage \( V_s = 0.65 \, \text{V} \).

### Solution

The total forward-bias current density is given by

\[
J = J_s \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right]
\]

We determined the reverse-saturation current density in Example 8.2, so we can write

\[
J = (4.155 \times 10^{-11}) \left[ \exp \left( \frac{0.65}{0.0259} \right) - 1 \right] = 3.295 \, \text{A/cm}^2
\]

The total current far from the junction in the n region will be majority carrier electron drift current, so we can write

\[
J = J_n \approx e \mu_n N_d E
\]

The doping concentration is \( N_d = 10^{16} \, \text{cm}^{-3} \), and, if we assume \( \mu_n = 1350 \, \text{cm}^2/\text{V-s} \), then the electric field must be

\[
E = \frac{J_n}{e \mu_n N_d} = \frac{3.295}{(1.6 \times 10^{-19})(1350)(10^{16})} = 1.525 \, \text{V/cm}
\]

### Comment

We assumed, in the derivation of the current–voltage equation, that the electric field in the neutral p and n regions was zero. Although the electric field is not zero, this example shows that the magnitude is very small—thus the approximation of zero electric field is very good.

### EXERCISE PROBLEM

Ex 8.4 Determine the electric field in the neutral n region and neutral p region for the GaAs pn junction diode described in Ex 8.3.

## 8.1.7 Temperature Effects

The ideal reverse-saturation current density \( J_s \), given by Equation (8.26), is a function of the thermal-equilibrium minority carrier concentrations \( n_{p0} \) and \( p_{n0} \). These minority carrier concentrations are proportional to \( n_i^2 \), which is a very strong function of temperature. For a silicon pn junction, the ideal reverse-saturation current density will increase by approximately a factor of 4 for every \( 10^\circ \text{C} \) increase in temperature.

# 8.1 pn Junction Current

The forward-bias current–voltage relation is given by Equation (8.27). This relation includes \( J_s \) as well as the \(\exp(eV/kT)\) factor, making the forward-bias current–voltage relation a function of temperature also. As temperature increases, less forward-bias voltage is required to obtain the same diode current. If the voltage is held constant, the diode current will increase as temperature increases. The change in forward-bias current with temperature is less sensitive than the reverse-saturation current.

## Objective

Determine the change in the forward-bias voltage on a pn junction with a change in temperature to maintain a constant diode current.

Consider a silicon pn junction initially biased at 0.60 V at \( T = 300 \, K \). Assume the temperature increases to \( T = 310 \, K \). Calculate the change in the forward-bias voltage required to maintain a constant current through the junction.

### Solution

The forward-bias current can be written as follows:

\[
J \approx \exp\left(-\frac{E_g}{kT}\right) \exp\left(\frac{eV_a}{kT}\right)
\]

If the temperature changes, we may take the ratio of the diode currents at the two temperatures. This ratio is

\[
\frac{J_2}{J_1} = \frac{\exp\left(-\frac{E_g}{kT_2}\right) \exp\left(\frac{eV_{a2}}{kT_2}\right)}{\exp\left(-\frac{E_g}{kT_1}\right) \exp\left(\frac{eV_{a1}}{kT_1}\right)}
\]

If current is to be held constant, then \( J_1 = J_2 \), and we must have

\[
\frac{E_g - eV_{a2}}{kT_2} = \frac{E_g - eV_{a1}}{kT_1}
\]

For \( T_1 = 300 \, K, \, T_2 = 310 \, K, \, E_g = 1.12 \, eV, \) and \( V_{a1} = 0.60 \, V \). We then find

\[
\frac{1.12 - V_{a2}}{310} = \frac{1.12 - 0.60}{300}
\]

which yields

\[
V_{a2} = 0.5827 \, V
\]

### Comment

The change in the forward-bias voltage is \(-17.3 \, mV\) for a \(10^\circ C\) temperature change.

### EXERCISE PROBLEM

**Ex 8.5** Repeat Example 8.5 for a GaAs pn junction diode biased at \( V_a = 1.050 \, V \) for \( T = 300 \, K \).

----

## 8.1.8 The “Short” Diode

We assumed in the previous analysis that both p and n regions were long compared with the minority carrier diffusion lengths. In many pn junction structures, one region may, in fact, be short compared with the minority carrier diffusion length. Figure 8.11 shows one such example: the length \( W_n \) is assumed to be much smaller than the minority carrier hole diffusion length, \( L_p \).

!Geometry of a "short" diode.

The steady-state excess minority carrier hole concentration in the n region is determined from Equation (8.9), which is given as

\[
\frac{d^2 (\delta p_n)}{dx^2} - \frac{\delta p_n}{L_p^2} = 0
\]

The original boundary condition at \( x = x_n \) still applies, given by Equation (8.11a) as

\[
p_n(x_n) = p_{n0} \exp \left( \frac{eV_a}{kT} \right)
\]

A second boundary condition needs to be determined. In many cases we assume that an ohmic contact exists at \( x = (x_n + W_n) \), implying an infinite surface-recombination velocity and therefore an excess minority carrier concentration of zero. The second boundary condition is then written as

\[
p_n(x = x_n + W_n) = p_{n0} \tag{8.30}
\]

The general solution to Equation (8.9) is again given by Equation (8.12), which was

\[
\delta p_n (x) = p_n(x) - p_{n0} = A e^{x/L_p} + B e^{-x/L_p} \quad (x \geq x_n)
\]

In this case, because of the finite length of the n region, both terms of the general solution must be retained. Applying the boundary conditions of Equations (8.11b) and (8.30), the excess minority carrier concentration is given by

\[
\delta p_n(x) = p_{n0} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] \frac{\sinh \left[ (x_n + W_n - x)/L_p \right]}{\sinh[W_n/L_p]} \tag{8.31}
\]

Equation (8.31) is the general solution for the excess minority carrier hole concentration in the n region of a forward-biased pn junction. If \( W_n \gg L_p \), the assumption for the long diode, Equation (8.31) reduces to the previous result given by Equation (8.14). If \( W_n \ll L_p \), we can approximate the hyperbolic sine terms by

\[
\sinh \left[ \frac{x_n + W_n - x}{L_p} \right] \approx \frac{x_n + W_n - x}{L_p} \tag{8.32a}
\]

and

\[
\sinh \left[ \frac{W_n}{L_p} \right] \approx \frac{W_n}{L_p} \tag{8.32b}
\]

Then Equation (8.31) becomes

\[
\delta p_n(x) = p_{n0} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] \frac{x_n + W_n - x}{W_n} \tag{8.33}
\]

```markdown
The minority carrier concentration becomes a linear function of distance. The minority carrier hole diffusion current density is given by

\[
J_p = -eD_p \frac{d\Delta p(x)}{dx}
\]

so that in the short n region, we have

\[
J_p(x) = \frac{eD_p \Delta p_0}{W_n} \left[ \exp \left( \frac{eV_A}{kT} \right) - 1 \right]
\]

(8.34)

The minority carrier hole diffusion current density now contains the length \( W_n \) in the denominator, rather than the diffusion length \( L_p \). The diffusion current density is larger for a short diode than for a long diode since \( W_n \ll L_p \). In addition, since the minority carrier concentration is approximately a linear function of distance through the n region, the minority carrier diffusion current density is a constant. This constant current implies that there is no recombination of minority carriers in the short region.

----

### TEST YOUR UNDERSTANDING

**TYU 8.1**  
The doping concentrations in a GaAs pn junction diode at \( T = 300 \, \text{K} \) are \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). The minority carrier concentration at either space charge edge is to be no larger than 10 percent of the respective majority carrier concentration. Calculate the maximum forward-bias voltage that may be applied to this junction and still meet the required specifications.  
[\( \Delta L901 = (\text{exu}^9 \Delta) \text{suV} \)]

**TYU 8.2**  
A silicon pn junction at \( T = 300 \, \text{K} \) has the following parameters: \( N_d = 5 \times 10^{16} \, \text{cm}^{-3} \), \( N_a = 1 \times 10^{16} \, \text{cm}^{-3} \), \( D_n = 25 \, \text{cm}^2/\text{s} \), \( D_p = 10 \, \text{cm}^2/\text{s} \), \( \tau_n = 5 \times 10^{-7} \, \text{s} \), and \( \tau_p = 1 \times 10^{-7} \, \text{s} \). The cross-sectional area is \( A = 10^{-2} \, \text{cm}^2 \) and the forward-bias voltage is \( V_A = 0.625 \, \text{V} \). Calculate the (a) minority electron diffusion current at the space charge edge, (b) minority hole diffusion current at the space charge edge, and (c) total current in the pn junction diode.  
[\( \nuU \tau T (q) \nuU 601 (q) \nuU 1510 (q) \nuU \)]

**TYU 8.3**  
Consider the silicon pn junction diode described in TYU 8.2. The p region is long and the n region is short with \( W_n = 2 \, \mu\text{m} \). (a) Calculate the electron and hole currents in the depletion region. (b) Why has the hole current increased compared to that found in TYU 8.2?  
[\( \text{paseau} \nuU \tau T (q) \nuU 1510 (q) \nuU 175 (q) \nuU 175 = q \nuU 1510 = \nu (q) \nuU \)]

----

## 8.2 | GENERATION–RECOMBINATION CURRENTS AND HIGH-INJECTION LEVELS

In the derivation of the ideal current–voltage relationship, we assumed low injection and neglected any effects occurring within the space charge region. High-level injection and other current components generated within the space charge region cause
```

### 8.2.1 Generation–Recombination Currents

The recombination rate of excess electrons and holes, given by the Shockley–Read–Hall recombination theory, was written as

\[
R = \frac{C_n C_p N_t (np - n_i^2)}{C_n (n + n') + C_p (p + p')}
\]

(8.35)

The parameters \( n \) and \( p \) are, as usual, the concentrations of electrons and holes, respectively.

**Reverse-Biased Generation Current**  
For a pn junction under reverse bias, we have argued that the mobile electrons and holes have essentially been swept out of the space charge region. Accordingly, within the space charge region, \( n \approx p \approx 0 \). The recombination rate from Equation (8.35) becomes

\[
R = \frac{-C_n C_p N_t n_i^2}{C_n n' + C_p p'}
\]

(8.36)

The negative sign implies a negative recombination rate; hence, we are really generating electron–hole pairs within the reverse-biased space charge region. The recombination of excess electrons and holes is the process whereby we are trying to reestablish thermal equilibrium. Since the concentration of electrons and holes is essentially zero within the reverse-biased space charge region, electrons and holes are being generated via the trap level to also try to reestablish thermal equilibrium. This generation process is schematically shown in Figure 8.12. As the electrons and holes

!Figure 8.12

**Figure 8.12** | Generation process in a reverse-biased pn junction.

### 8.2 Generation–Recombination Currents and High-Injection Levels

are generated, they are swept out of the space charge region by the electric field. The flow of charge is in the direction of a reverse-biased current. This **reverse-biased generation current**, caused by the generation of electrons and holes in the space charge region, is in addition to the ideal reverse-biased saturation current.

We may calculate the density of the reverse-biased generation current by considering Equation (8.36). If we make a simplifying assumption and let the trap level be at the intrinsic Fermi level, then from Equations (6.92) and (6.97), we have that \( n' = n_i \) and \( p' = n_i \). Equation (8.36) now becomes

\[
R = \frac{-n_i}{1 + \frac{1}{N_C \tau_p} + \frac{1}{N_V \tau_n}}
\]

(8.37)

Using the definitions of lifetimes from Equations (6.103) and (6.104), we may write Equation (8.37) as

\[
R = \frac{-n_i}{\tau_{p0} + \tau_{n0}}
\]

(8.38)

If we define a new lifetime as the average of \( \tau_{p0} \) and \( \tau_{n0} \), or

\[
\tau_0 = \frac{\tau_{p0} + \tau_{n0}}{2}
\]

(8.39)

then the recombination rate can be written as

\[
R = \frac{-n_i}{2\tau_0} = -G
\]

(8.40)

The negative recombination rate implies a generation rate, so \( G \) is the generation rate of electrons and holes in the space charge region.

The generation current density may be determined from

\[
J_{gen} = \int_0^W eG \, dx
\]

(8.41)

where the integral is over the space charge region. If we assume that the generation rate is constant throughout the space charge region, then we obtain

\[
J_{gen} = \frac{en_i W}{2\tau_0}
\]

(8.42)

The total reverse-biased current density is the sum of the ideal reverse saturation current density and the generation current density, or

\[
J_r = J_s + J_{gen}
\]

(8.43)

The ideal reverse-saturation current density \( J_s \) is independent of the reverse-biased voltage. However, \( J_{gen} \) is a function of the depletion width \( W \), which in turn is a function of the reverse-biased voltage. The actual reverse-biased current density, then, is no longer independent of the reverse-biased voltage.

## EXAMPLE 8.6

**Objective:** Determine the relative magnitudes of the ideal reverse-saturation current density and the generation current density in a reverse-biased pn junction.

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with parameters \( D_n = 25 \, \text{cm}^2/\text{s}, D_p = 10 \, \text{cm}^2/\text{s}, N_d = N_a = 10^{16} \, \text{cm}^{-3} \), and \( \tau_n = \tau_p = \tau_0 = 5 \times 10^{-7} \, \text{s} \). Assume the diode is reverse biased at \( V_R = 5 \, \text{V} \).

### Solution

The ideal reverse-saturation current density was calculated in Example 8.2 and was found to be \( J_0 = 4.155 \times 10^{-11} \, \text{A/cm}^2 \).

The built-in potential is found as

\[
V_{bi} = V_T \ln \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left( \frac{(10^{16})(10^{16})}{(1.5 \times 10^{10})^2} \right) = 0.695 \, \text{V}
\]

The depletion width is found to be

\[
W = \left[ \frac{2 \varepsilon_s (V_{bi} + V_R)}{e} \left( \frac{N_a + N_d}{N_a N_d} \right) \right]^{1/2}
\]

\[
= \left[ \frac{(2)(11.7)(8.85 \times 10^{-14})(0.695 + 5)}{1.6 \times 10^{-19}} \right] \left[ \frac{10^{16} + 10^{16}}{(10^{16})(10^{16})} \right]^{1/2}
\]

\[
= 1.214 \times 10^{-4} \, \text{cm}
\]

The generation current density is then found to be

\[
J_{gen} = \frac{e n_i W}{2 \tau_0} = \frac{(1.6 \times 10^{-19})(1.5 \times 10^{10})(1.214 \times 10^{-4})}{2(5 \times 10^{-7})}
\]

or

\[
J_{gen} = 2.914 \times 10^{-7} \, \text{A/cm}^2
\]

The ratio of the two currents is

\[
\frac{J_{gen}}{J_0} = \frac{2.914 \times 10^{-7}}{4.155 \times 10^{-11}} = 7 \times 10^3
\]

### Comment

Comparing the solutions for the two current densities, it is obvious that, for the silicon pn junction diode at room temperature, the generation current density is approximately four orders of magnitude larger than the ideal saturation current density. The generation current is the dominant reverse-biased current in a silicon pn junction diode.

### EXERCISE PROBLEM

**Ex 8.6** Consider a GaAs pn junction diode at \( T = 300 \, \text{K} \) with parameters \( N_d = 8 \times 10^{16} \, \text{cm}^{-3}, N_a = 2 \times 10^{16} \, \text{cm}^{-3}, D_n = 207 \, \text{cm}^2/\text{s}, D_p = 9.80 \, \text{cm}^2/\text{s}, \) and \( \tau_n = \tau_p = \tau_0 = 5 \times 10^{-8} \, \text{s} \). (a) Calculate the ideal reverse-biased saturation current density. (b) Find the reverse-biased generation current density if the diode is reverse biased at \( V_R = 5 \, \text{V} \). (c) Determine the ratio of \( J_{gen} \) to \( J_0 \).

# 8.2 Generation–Recombination Currents and High-Injection Levels

## Forward-Bias Recombination Current

For the reverse-biased pn junction, electrons and holes are essentially completely swept out of the space charge region so that \( n = p \approx 0 \). Under forward bias, however, electrons and holes are injected across the space charge region, so we do, in fact, have some excess carriers in the space charge region. The possibility exists that some of these electrons and holes will recombine within the space charge region and not become part of the minority carrier distribution.

The recombination rate of electrons and holes is again given from Equation (8.35) as

\[
R = \frac{C_n C_p N_i (np - n_i^2)}{C_n (n + n') + C_p (p + p')}
\]

Dividing both numerator and denominator by \( C_n C_p N_i \) and using the definitions of \( \tau_{n0} \) and \( \tau_{p0} \), we may write the recombination rate as

\[
R = \frac{np - n_i^2}{\tau_{p0} (n + n') + \tau_{n0} (p + p')} \tag{8.44}
\]

Figure 8.13 shows the energy-band diagram of the forward-biased pn junction. Shown in the figure are the intrinsic Fermi level and the quasi-Fermi levels for electrons and holes. From the results of Chapter 6, we may write the electron concentration as

\[
n = n_i \exp \left[ \frac{E_{Fn} - E_{Fi}}{kT} \right] \tag{8.45}
\]

and the hole concentration as

\[
p = n_i \exp \left[ \frac{E_{Fi} - E_{Fp}}{kT} \right] \tag{8.46}
\]

where \( E_{Fn} \) and \( E_{Fp} \) are the quasi-Fermi levels for electrons and holes, respectively.

!Energy-band diagram

**Figure 8.13** | Energy-band diagram of a forward-biased pn junction including quasi-Fermi levels.

```markdown
# CHAPTER 8 The pn Junction Diode

!Figure 8.14

**Figure 8.14** Relative magnitude of the recombination rate through the space charge region of a forward-biased pn junction.

From Figure 8.13, we may note that

\[
(E_{Fn} - E_{Fi}) + (E_{Fp} - E_{Fi}) = eV_a
\]

(8.47)

where \( V_a \) is the applied forward-bias voltage. Again, if we assume that the trap level is at the intrinsic Fermi level, then \( n' = p' = n_i \). Figure 8.14 shows a plot of the relative magnitude of the recombination rate as a function of distance through the space charge region. This plot was generated using Equations (8.44), (8.45), (8.46), and (8.47). A very sharp peak occurs at the metallurgical junction (\( x = 0 \)).

At the center of the space charge region, we have

\[
E_{Fn} - E_{Fi} = E_{Fi} - E_{Fp} = \frac{eV_a}{2}
\]

(8.48)

Equations (8.45) and (8.46) then become

\[
n = n_i \exp \left( \frac{eV_a}{2kT} \right)
\]

(8.49)

and

\[
p = n_i \exp \left( \frac{eV_a}{2kT} \right)
\]

(8.50)

If we assume that \( n' = p' = n_i \) and that \( \tau_{n0} = \tau_{p0} = \tau_0 \), then Equation (8.44) becomes

\[
R_{max} = \frac{n_i}{2\tau_0} \left[ \frac{\exp(eV_a/kT) - 1}{\exp(eV_a/2kT) + 1} \right]
\]

(8.51)

which is the maximum recombination rate for electrons and holes that occurs at the center of the forward-biased pn junction. If we assume that \( V_a \gg kT/e \), we may
```

## 8.2 Generation–Recombination Currents and High-Injection Levels

neglect the \((-1)\) term in the numerator and the \((+1)\) term in the denominator. Equation (8.51) then becomes

\[
R_{\text{max}} = \frac{n_i}{2\tau_0} \exp\left(\frac{eV_a}{2kT}\right)
\]

(8.52)

The recombination current density may be calculated from

\[
J_{\text{rec}} = \int_0^W eR \, dx
\]

(8.53)

where again the integral is over the entire space charge region. In this case, however, the recombination rate is not a constant through the space charge region. We have calculated the maximum recombination rate at the center of the space charge region, so we may write

\[
J_{\text{rec}} = ex' \frac{n_i}{2\tau_0} \exp\left(\frac{eV_a}{2kT}\right)
\]

(8.54)

where \(x'\) is a length over which the maximum recombination rate is effective. However, since \(\tau_0\) may not be a well-defined or known parameter, it is customary to write

\[
J_{\text{rec}} = \frac{eWn_i}{2\tau_0} \exp\left(\frac{eV_a}{2kT}\right) = J_0 \exp\left(\frac{eV_a}{2kT}\right)
\]

(8.55)

where \(W\) is the space charge width.

### Total Forward-Bias Current

The total forward-bias current density in the pn junction is the sum of the recombination and the ideal diffusion current densities. Figure 8.15 shows a plot of the minority carrier hole concentration in the neutral

!Figure 8.15

**Figure 8.15** Because of recombination, additional holes from the p region must be injected into the space charge region to establish the minority carrier hole concentration in the n region.

# Chapter 8: The pn Junction Diode

n region. This distribution yields the ideal hole diffusion current density and is a function of the minority carrier hole diffusion length and the applied junction voltage. The distribution is established as a result of holes being injected across the space charge region. If, now, some of the injected holes in the space charge region are lost due to recombination, then additional holes must be injected from the p region to make up for this loss. The flow of these additional injected carriers, per unit time, results in the recombination current. This added component is schematically shown in the figure.

The total forward-bias current density is the sum of the recombination and the ideal diffusion current densities, so we can write

\[
J = J_{\text{rec}} + J_D
\]

(8.56)

where \( J_{\text{rec}} \) is given by Equation (8.55) and \( J_D \) is given by

\[
J_D = J_s \exp\left(\frac{eV_a}{kT}\right)
\]

(8.57)

The (−1) term in Equation (8.27) has been neglected. The parameter \( J_s \) is the ideal reverse-saturation current density, and from previous discussion, the value of \( J_{a} \) from the recombination current is larger than the value of \( J_s \).

If we take the natural log of Equations (8.55) and (8.57), we obtain

\[
\ln J_{\text{rec}} = \ln J_{a} + \frac{eV_a}{2kT} = \ln J_{a} + \frac{V_a}{2V_t}
\]

(8.58a)

and

\[
\ln J_D = \ln J_s + \frac{eV_a}{kT} = \ln J_s + \frac{V_a}{V_t}
\]

(8.58b)

Figure 8.16 shows the recombination and diffusion current components plotted on a log current scale as a function of \( V_a/V_t \). The slopes of the two curves are not the same. Also shown in the figure is the total current density—the sum of the two current components. We may notice that, at a low current density, the recombination current dominates, and at a higher current density, the ideal diffusion current dominates.

In general, the diode current–voltage relationship may be written as

\[
I = I_s \left[\exp\left(\frac{eV_a}{nkT}\right) - 1\right]
\]

(8.59)

where the parameter \( n \) is called the **ideality factor**. For a large forward-bias voltage, \( n \approx 1 \) when diffusion dominates, and for low forward-bias voltage, \( n \approx 2 \) when recombination dominates. There is a transition region where \( 1 < n < 2 \).

## 8.2.2 High-Level Injection

In the derivation of the ideal diode \( I-V \) relationship, we assumed that low injection was valid. Low injection implies that the excess minority carrier concentrations are always much less than the majority carrier concentration.

# 8.2 Generation–Recombination Currents and High-Injection Levels

!Graph

**Figure 8.16** Ideal diffusion, recombination, and total current in a forward-biased pn junction.

However, as the forward-bias voltage increases, the excess carrier concentrations increase and may become comparable or even greater than the majority carrier concentration. From Equation (8.18), we can write

\[
np = n_i^2 \exp\left(\frac{V_a}{V_t}\right)
\]

We have that \( n = n_0 + \delta n \) and \( p = p_0 + \delta p \), so that

\[
(n_0 + \delta n)(p_0 + \delta p) = n_i^2 \exp\left(\frac{V_a}{V_t}\right) \tag{8.60}
\]

Under high-level injection, we may have \( \delta n > n_0 \) and \( \delta p > p_0 \) so that Equation (8.60) becomes approximately

\[
(\delta n)(\delta p) \approx n_i^2 \exp\left(\frac{V_a}{V_t}\right) \tag{8.61}
\]

Since \( \delta n = \delta p \), then

\[
\delta n = \delta p \approx n_i \exp\left(\frac{V_a}{2V_t}\right) \tag{8.62}
\]

The diode current is proportional to the excess carrier concentration so that, under high-level injection, we have

\[
I \propto \exp\left(\frac{V_a}{2V_t}\right) \tag{8.63}
\]

# CHAPTER 8 The pn Junction Diode

!Figure 8.17

**Figure 8.17** | Forward-bias current versus voltage from low forward bias to high forward bias.

In the high-level injection region, it takes a larger increase in diode voltage to produce a given increase in diode current.

The diode forward-bias current, from low-bias levels to high-bias levels, is plotted in Figure 8.17. This plot shows the effect of recombination at low-bias voltages and high-level injection at high-bias voltages.

----

**TEST YOUR UNDERSTANDING**

**TYU 8.4** Consider a silicon pn junction diode at \( T = 300 \, K \) with parameters \( N_s = 2 \times 10^{15} \, \text{cm}^{-3} \), \( N_i = 8 \times 10^{16} \, \text{cm}^{-3} \), \( D_n = 10 \, \text{cm}^2/\text{s} \), \( D_p = 25 \, \text{cm}^2/\text{s} \), and \( \tau_p = \tau_n = 10^{-7} \, \text{s} \). The diode is forward biased at \( V_a = 0.35 \, \text{V} \). (a) Calculate the ideal diode current density. (b) Find the forward-biased recombination current density. (c) Determine the ratio of recombination current to the ideal diffusion current.

\[ \text{Use: } \frac{I_s}{q} \cdot \frac{W}{V_t} \cdot 0.1 \times 0.025 \cdot \frac{q}{W} \cdot 0.1 \times \frac{L}{T} \cdot \frac{V}{W} \]

----

## 8.3 | SMALL-SIGNAL MODEL OF THE pn JUNCTION

We have been considering the dc characteristics of the pn junction diode. When semiconductor devices with pn junctions are used in linear amplifier circuits, for example, sinusoidal signals are superimposed on the dc currents and voltages, so that the small-signal characteristics of the pn junction become important.

# 8.3.1 Diffusion Resistance

The ideal current–voltage relationship of the pn junction diode was given by Equation (8.27), where \( J \) and \( J_s \) are current densities. If we multiply both sides of the equation by the junction cross-sectional area, we have

\[
I_D = I_s \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.64)

where \( I_D \) is the diode current and \( I_s \) is the diode reverse-saturation current.

Assume that the diode is forward-biased with a dc voltage \( V_0 \) producing a dc diode current \( I_{DQ} \). If we now superimpose a small, low-frequency sinusoidal voltage as shown in Figure 8.18, then a small sinusoidal current will be produced, superimposed on the dc current. The ratio of sinusoidal current to sinusoidal voltage is called the incremental conductance. In the limit of a very small sinusoidal current and voltage, the small-signal incremental conductance is just the slope of the dc current–voltage curve, or

\[
g_d = \left. \frac{dI_D}{dV_a} \right|_{I_D=I_{DQ}}
\]

(8.65)

The reciprocal of the incremental conductance is the incremental resistance, defined as

\[
r_d = \left. \frac{dV_a}{dI_D} \right|_{I_D=I_{DQ}}
\]

(8.66)

where \( I_{DQ} \) is the dc quiescent diode current.

!Figure 8.18

**Figure 8.18** Curve showing the concept of the small-signal diffusion resistance.

# Chapter 8: The pn Junction Diode

If we assume that the diode is biased sufficiently far in the forward-bias region, then the \((-1)\) term can be neglected and the incremental conductance becomes

\[
g_d = \frac{dI}{dV} \bigg|_{V=V_o} = \left( \frac{e}{kT} \right) I_o \exp \left( \frac{eV_o}{kT} \right) \approx \frac{I_{DQ}}{V_T}
\]

(8.67)

The small-signal incremental resistance is then the reciprocal function, or

\[
r_d = \frac{V_T}{I_{DQ}}
\]

(8.68)

The incremental resistance decreases as the bias current increases, and is inversely proportional to the slope of the \(I-V\) characteristic as shown in Figure 8.18. The incremental resistance is also known as the **diffusion resistance**.

## 8.3.2 Small-Signal Admittance

In the last chapter, we considered the pn junction capacitance as a function of the reverse-biased voltage. When the pn junction diode is forward-biased, another capacitance becomes a factor in the diode admittance. The small-signal admittance, or impedance, of the pn junction under forward bias is derived using the minority carrier diffusion current relations we have already considered.

**Qualitative Analysis** Before we delve into the mathematical analysis, we can qualitatively understand the physical processes that lead to a diffusion capacitance, which is one component of the junction admittance. Figure 8.19a schematically shows a pn junction forward biased with a dc voltage. A small ac voltage is also applied.

!Figure 8.19

**Figure 8.19** (a) A pn junction with an ac voltage superimposed on a forward-biased dc value; (b) the hole concentration versus time at the space charge edge; (c) the hole concentration versus distance in the n region at three different times.

- **(a)** Schematic of pn junction with ac voltage.
- **(b)** Hole concentration vs. time.
- **(c)** Hole concentration vs. distance.

- \(V_{ac} = \Phi \sin \omega t\)
- \(p_{n0} \exp \left( \frac{V_{ac} + \Phi}{V_T} \right)\)
- \(p_{n0} \exp \left( \frac{V_{ac} - \Phi}{V_T} \right)\)
- \(x = 0\)
- Time: \(t = t_0, t_1, t_2\)

### 8.3 Small-Signal Model of the pn Junction

superimposed on the dc voltage so that the total forward-biased voltage can be written as \( V_a = V_{dc} + v(t) \) on set.

As the voltage across the junction changes, the number of holes injected across the space charge region into the n region also changes. Figure 8.19b shows the hole concentration at the space charge edge as a function of time. At \( t = t_0 \), the ac voltage is zero so that the concentration of holes at \( x = 0 \) is just given by \( p_n(0) = p_{n0} \exp(V_a/V_T) \), which is what we have seen previously.

Now, as the ac voltage increases during its positive half cycle, the concentration of holes at \( x = 0 \) will increase and reach a peak value at \( t = t_1 \), which corresponds to the peak value of the ac voltage. When the ac voltage is on its negative half cycle, the total voltage across the junction decreases so that the concentration of holes at \( x = 0 \) decreases. The concentration reaches a minimum value at \( t = t_2 \), which corresponds to the time that the ac voltage reaches its maximum negative value. The minority carrier hole concentration at \( x = 0 \), then, has an ac component superimposed on the dc value as indicated in Figure 8.19b.

As previously discussed, the holes at the space charge edge (\( x = 0 \)) diffuse into the n region where they recombine with the majority carrier electrons. We assume that the period of the ac voltage is large compared to the time it takes carriers to diffuse into the n region. The hole concentration as a function of distance into the n region can then be treated as a steady-state distribution. Figure 8.19c shows the steady-state hole concentrations at three different times. At \( t = t_0 \), the ac voltage is zero, so the \( t = t_0 \) curve corresponds to the hole distribution established by the dc voltage. The \( t = t_1 \) curve corresponds to the distribution established when the ac voltage has reached its peak positive value, and the \( t = t_2 \) curve corresponds to the distribution established when the ac voltage has reached its maximum negative value. The shaded areas represent the charge \( \Delta Q \) that is alternately charged and discharged during the ac voltage cycle.

Exactly the same process is occurring in the p region with the electron concentration. The mechanism of charging and discharging of holes in the n region and electrons in the p region leads to a capacitance. This capacitance is called diffusion capacitance. The physical mechanism of this diffusion capacitance is different from that of the junction capacitance discussed in the last chapter. We show that the magnitude of the diffusion capacitance in a forward-biased pn junction is usually substantially larger than the junction capacitance.

#### Mathematical Analysis

The minority carrier distribution in the pn junction will be derived for the case when a small sinusoidal voltage is superimposed on the dc junction voltage. We can then determine small signal, or ac, diffusion currents from these minority carrier functions. Figure 8.20 shows the minority carrier distribution in a pn junction when a forward-biased dc voltage is applied. The origin, \( x = 0 \), is set at the edge of the space charge region on the n side for convenience. The minority carrier hole concentration at \( x = 0 \) is given by Equation (8.7) as \( p_n(0) = p_{n0} \exp(eV_a/kT) \), where \( V_a \) is the applied voltage across the junction.

Now let

\[
V_a = V_0 + v(t)
\]

(8.69)

# Chapter 8 The pn Junction Diode

!Figure 8.20

**Figure 8.20** The dc characteristics of a forward-biased pn junction used in the small-signal admittance calculations.

where \( V_0 \) is the dc quiescent bias voltage and \( v_l(t) \) is the ac signal voltage that is superimposed on this dc level. We may now write

\[
p_n(x = 0) = p_{n0} \exp \left[ \frac{e(V_0 + v_l(t))}{kT} \right] = p_n(0, t)
\]

(8.70)

Equation (8.70) may be written as

\[
p_n(0, t) = p_{n0} \exp \left[ \frac{ev_l(t)}{kT} \right]
\]

(8.71)

where

\[
p_{n0} = p_{n0} \exp \left( \frac{eV_0}{kT} \right)
\]

(8.72)

If we assume that \( |v_l(t)| \ll (kT/e) = V_T \), then the exponential term in Equation (8.71) may be expanded into a Taylor series retaining only the linear terms, and the minority carrier hole concentration at \( x = 0 \) can be written as

\[
p_n(0, t) \approx p_{n0} \left[ 1 + \frac{v_l(t)}{V_T} \right]
\]

(8.73)

If we assume that the time-varying voltage \( v_l(t) \) is a sinusoidal signal, we can write Equation (8.73) as

\[
p_n(0, t) = p_{n0} \left[ 1 + \frac{\hat{V}_l}{V_T} e^{j\omega t} \right]
\]

(8.74)

where \( \hat{V}_l \) is the phasor of the applied sinusoidal voltage. Equation (8.74) will be used as the boundary condition in the solution of the time-dependent diffusion equation for the minority carrier holes in the n region.

In the neutral n region (\( x > 0 \)), the electric field is assumed to be zero; thus, the behavior of the excess minority carrier holes is determined from the equation

\[
D_p \frac{\partial^2 (\delta p_n)}{\partial x^2} - \frac{\delta p_n}{\tau_{p0}} = \frac{\partial (\delta p_n)}{\partial t}
\]

(8.75)

# 8.3 Small-Signal Model of the pn Junction

where \( \delta p_n \) is the excess hole concentration in the n region. We are assuming that the ac signal voltage \( v_i (t) \) is sinusoidal. We then expect the steady-state solution for \( \delta p_n \) to be of the form of a sinusoidal solution superimposed on the dc solution, or

\[
\delta p_n(x, t) = \delta p_0(x) + p_1(x)e^{j\omega t}
\]

(8.76)

where \( \delta p_0(x) \) is the dc excess carrier concentration and \( p_1(x) \) is the magnitude of the ac component of the excess carrier concentration. The expression for \( \delta p_0(x) \) is the same as that given in Equation (8.14).

Substituting Equation (8.76) into the differential Equation (8.75), we obtain

\[
D_p \left[ \frac{\partial^2 [\delta p_0(x)]}{\partial x^2} + \frac{\partial^2 p_1(x)}{\partial x^2} e^{j\omega t} \right] - \frac{\delta p_0(x) + p_1(x)e^{j\omega t}}{\tau_0} = j\omega p_1(x)e^{j\omega t}
\]

(8.77)

We may rewrite this equation, combining the time-dependent and time-independent terms, as

\[
\left[ D_p \frac{\partial^2 [\delta p_0(x)]}{\partial x^2} - \frac{\delta p_0(x)}{\tau_0} \right] + \left[ D_p \frac{\partial^2 p_1(x)}{\partial x^2} - \frac{p_1(x)}{\tau_0} - j\omega p_1(x) \right] e^{j\omega t} = 0
\]

(8.78)

If the ac component, \( p_1(x) \), is zero, then the first bracketed term is just the differential Equation (8.10), which is identically zero. Then we have, from the second bracketed term,

\[
D_p \frac{d^2 p_1(x)}{dx^2} - \frac{p_1(x)}{\tau_0} - j\omega p_1(x) = 0
\]

(8.79)

Noting that \( L_p^2 = D_p \tau_0 \), Equation (8.79) may be rewritten in the form

\[
\frac{d^2 p_1(x)}{dx^2} - \frac{(1 + j\omega \tau_0)}{L_p^2} p_1(x) = 0
\]

(8.80)

or

\[
\frac{d^2 p_1(x)}{dx^2} - C_p^2 p_1(x) = 0
\]

(8.81)

where

\[
C_p^2 = \frac{(1 + j\omega \tau_0)}{L_p^2}
\]

(8.82)

The general solution to Equation (8.81) is

\[
p_1(x) = K_1 e^{-C_p x} + K_2 e^{C_p x}
\]

(8.83)

One boundary condition is that \( p_1(x \to + \infty) = 0 \), which implies that the coefficient \( K_2 = 0 \). Then

\[
p_1(x) = K_1 e^{-C_p x}
\]

(8.84)

Applying the boundary condition at \( x = 0 \) from Equation (8.74), we obtain

\[
p_1(0) = K_1 = p_{n0} \left( \frac{V_i}{V_t} \right)
\]

(8.85)

# CHAPTER 8 The pn Junction Diode

The hole diffusion current density can be calculated at \( x = 0 \). This is given by

\[
J_p = -eD_p \frac{\partial p}{\partial x} \bigg|_{x=0}
\]

(8.86)

If we consider a homogeneous semiconductor, the derivative of the hole concentration will be just the derivative of the excess hole concentration. Then

\[
J_p = -eD_p \frac{\partial (\delta p)}{\partial x} \bigg|_{x=0} = -eD_p \frac{\partial [\delta p(x)]}{\partial x} \bigg|_{x=0} = -eD_p \frac{\partial p(x)}{\partial x} \bigg|_{x=0} e^{j\omega t}
\]

(8.87)

We can write this equation in the form

\[
J_p = J_{p_0} + j_p(t)
\]

(8.88)

where

\[
J_{p_0} = -eD_p \frac{\partial [\delta p(x)]}{\partial x} \bigg|_{x=0} = \frac{eD_p p_{n_0}}{L_p} \left[ \exp \left( \frac{eV_0}{kT} \right) - 1 \right]
\]

(8.89)

Equation (8.89) is the dc component of the hole diffusion current density and is exactly the same as in the ideal I–V relation derived previously.

The sinusoidal component of the diffusion current density is then found from

\[
j_p(t) = \hat{j}_p e^{j\omega t} = -eD_p \frac{\partial p(x)}{\partial x} \bigg|_{x=0} e^{j\omega t}
\]

(8.90)

where \(\hat{j}_p\) is the current density phasor. Combining Equations (8.90), (8.84), and (8.85), we have

\[
\hat{J}_p = -eD_p (A - C_p) \left[ \frac{\hat{V}_i}{V_T} \right] e^{-j\omega t} \bigg|_{x=0}
\]

(8.91)

We can write the total ac hole current phasor as

\[
\hat{I}_p = A \hat{j}_p = eAD_p C_p \frac{\hat{v}_i}{V_T}
\]

(8.92)

where \( A \) is the cross-sectional area of the pn junction. Substituting the expression for \( C_p \), we obtain

\[
\hat{I}_p = \frac{eAD_p p_{n_0}}{L_p} \sqrt{1 + j\omega \tau_0} \left( \frac{\hat{v}_i}{V_T} \right)
\]

(8.93)

If we define

\[
I_{p_0} = \frac{eAD_p p_{L}}{L_p} = \frac{eAD_p p_{n_0}}{L_p} \exp \left( \frac{eV_0}{kT} \right)
\]

(8.94)

then Equation (8.93) becomes

\[
\hat{I}_p = I_{p_0} \sqrt{1 + j\omega \tau_0} \left( \frac{\hat{v}_i}{V_T} \right)
\]

(8.95)

Going through the same type of analysis for the minority carrier electrons in the p region, we obtain

\[
\hat{I}_n = I_{n_0} \sqrt{1 + j\omega \tau_0} \left( \frac{\hat{v}_i}{V_T} \right)
\]

(8.96)

## 8.3 Small-Signal Model of the pn Junction

where

\[
I_{0} = \frac{eAD_{n}n_{p0}}{L_{n}} \exp \left( \frac{eV_{0}}{kT} \right)
\]

(8.97)

The total ac current phasor is the sum of \(\hat{i}_{n}\) and \(\hat{i}_{p}\). The pn junction admittance is the total ac current phasor divided by the ac voltage phasor, or

\[
Y = \frac{\hat{i}}{\hat{V}_{i}} = \left( \frac{1}{\hat{V}_{i}} \right) [I_{p0} \sqrt{1 + j\omega\tau_{p0}} + I_{n0} \sqrt{1 + j\omega\tau_{n0}}]
\]

(8.98)

There is not a linear, lumped, finite, passive, bilateral network that can be synthesized to give this admittance function. However, we may make the following approximations. Assume that

\[
\omega\tau_{p0} \ll 1
\]

(8.99a)

and

\[
\omega\tau_{n0} \ll 1
\]

(8.99b)

These two assumptions imply that the frequency of the ac signal is not too large. Then we may write

\[
\sqrt{1 + j\omega\tau_{p0}} \approx 1 + \frac{j\omega\tau_{p0}}{2}
\]

(8.100a)

and

\[
\sqrt{1 + j\omega\tau_{n0}} \approx 1 + \frac{j\omega\tau_{n0}}{2}
\]

(8.100b)

Substituting Equations (8.100a) and (8.100b) into the admittance Equation (8.98), we obtain

\[
Y = \left( \frac{1}{\hat{V}_{i}} \right) \left[ I_{p0} \left( 1 + \frac{j\omega\tau_{p0}}{2} \right) + I_{n0} \left( 1 + \frac{j\omega\tau_{n0}}{2} \right) \right]
\]

(8.101)

If we combine the real and imaginary portions, we get

\[
Y = \left( \frac{1}{\hat{V}_{i}} \right) (I_{p0} + I_{n0}) + j\omega \left( \frac{1}{2\hat{V}_{i}} \right) (I_{p0}\tau_{p0} + I_{n0}\tau_{n0})
\]

(8.102)

Equation (8.102) may be written in the form

\[
Y = g_{d} + j\omega C_{d}
\]

(8.103)

The parameter \(g_{d}\) is called the **diffusion conductance** and is given by

\[
g_{d} = \left( \frac{1}{\hat{V}_{i}} \right) (I_{p0} + I_{n0}) = \frac{I_{DQ}}{\hat{V}_{i}}
\]

(8.104)

where \(I_{DQ}\) is the dc bias current. Equation (8.104) is exactly the same conductance as we obtained previously in Equation (8.67). The parameter \(C_{d}\) is called the **diffusion capacitance** and is given by

\[
C_{d} = \left( \frac{1}{2\hat{V}_{i}} \right) (I_{p0}\tau_{p0} + I_{n0}\tau_{n0})
\]

(8.105)

!Figure 8.21  
**Figure 8.21** Minority carrier concentration changes with changing forward-bias voltage.

The physics of the diffusion capacitance may be seen in Figure 8.21. The dc values of the minority carrier concentrations are shown along with the changes due to the ac component of voltage. The \(\Delta Q\) charge is alternately being charged and discharged through the junction as the voltage across the junction changes. The change in the stored minority carrier charge as a function of the change in voltage is the diffusion capacitance. One consequence of the approximations \(\omega \tau_p \ll 1\) and \(\omega \tau_n \ll 1\) is that there are no “wiggles” in the minority carrier curves. The sinusoidal frequency is low enough so that the exponential curves are maintained at all times.

----

**EXAMPLE 8.7**

**Objective:** Calculate the small-signal admittance parameters of a pn junction diode.

This example is intended to give an indication of the magnitude of the diffusion capacitance as compared with the junction capacitance considered in the last chapter. The diffusion resistance will also be calculated. Assume that \(N_D \gg N_A\) so that \(p_{n0} \gg n_{p0}\). This assumption implies that \(I_{p0} \gg I_{n0}\). Let \(T = 300 \, \text{K}\), \(\tau_{p0} = 10^{-7} \, \text{s}\), and \(I_{p0} = I_0 = 1 \, \text{mA}\).

**Solution**

The diffusion capacitance, with these assumptions, is given by

\[
C_d \approx \left( \frac{1}{2V_T} \right) (I_{p0} \tau_{p0}) = \frac{1}{(2)(0.0259)} (10^{-7} \times 10^{-3}) = 1.93 \times 10^{-9} \, \text{F}
\]

The diffusion resistance is

\[
r_d = \frac{V_T}{I_{p0}} = \frac{0.0259 \, \text{V}}{1 \, \text{mA}} = 25.9 \, \Omega
\]

**Comment**

The value of 1.93 nF for the diffusion capacitance of a forward-biased pn junction is three to four orders of magnitude larger than the junction capacitance of the reverse-biased pn junction, which we calculated in Example 7.5.

```markdown
### 8.3 Small-Signal Model of the pn Junction

#### EXERCISE PROBLEM

**Ex 8.7** A silicon pn junction diode at \( T = 300 \, \text{K} \) has the following parameters: \( N_d = 8 \times 10^{16} \, \text{cm}^{-3}, N_a = 2 \times 10^{16} \, \text{cm}^{-3}, D_n = 25 \, \text{cm}^2/\text{s}, D_p = 10 \, \text{cm}^2/\text{s}, \tau_{n0} = 5 \times 10^{-7} \, \text{s}, \) and \( \tau_{p0} = 10^{-7} \, \text{s} \). The cross-sectional area is \( A = 10^{-3} \, \text{cm}^2 \). Determine the diffusion resistance and diffusion capacitance if the diode is forward biased at (a) \( V_a = 0.550 \, \text{V} \) and (b) \( V_a = 0.610 \, \text{V} \).

The diffusion capacitance tends to dominate the capacitance terms in a forward-biased pn junction. The small-signal diffusion resistance can be fairly small if the diode current is a fairly large value. As the diode current decreases, the diffusion resistance increases. We will consider the impedance of forward-biased pn junctions again when we discuss bipolar transistors.

### 8.3.3 Equivalent Circuit

The small-signal equivalent circuit of the forward-biased pn junction is derived from Equation (8.103). This circuit is shown in Figure 8.22a. We need to add the junction capacitance, which will be in parallel with the diffusion resistance and diffusion capacitance. The last element we add, to complete the equivalent circuit, is a series resistance. The neutral n and p regions have finite resistances so the actual pn junction will include a series resistance. The complete equivalent circuit is given in Figure 8.22b.

The voltage across the actual junction is \( V_a \) and the total voltage applied to the pn diode is given by \( V_{\text{app}} \). The junction voltage \( V_a \) is the voltage in the ideal current–voltage expression. We can write the expression

\[
V_{\text{app}} = V_a + I_r
\]

(8.106)

Figure 8.23 is a plot of the current–voltage characteristic from Equation (8.106) showing the effect of the series resistance. A larger applied voltage is required to

!Figure 8.22

**Figure 8.22** (a) Small-signal equivalent circuit of ideal forward-biased pn junction diode; (b) complete small-signal equivalent circuit of pn junction.
```

# CHAPTER 8 The pn Junction Diode

!Figure 8.23  
**Figure 8.23** | Forward-biased \( I \text{-} V \) characteristics of a pn junction diode showing the effect of series resistance.

achieve the same current value when a series resistance is included. In most diodes, the series resistance will be negligible. In some semiconductor devices with pn junctions, however, the series resistance will be in a feedback loop; in these cases, the resistance is multiplied by a gain factor and becomes non-negligible.

----

### TEST YOUR UNDERSTANDING

**TYU 8.5**  
A GaAs pn junction diode at \( T = 300 \, \text{K} \) has the same parameters given in Ex 8.7 except that \( D_n = 207 \, \text{cm}^2/\text{s} \) and \( D_p = 9.80 \, \text{cm}^2/\text{s} \). Determine the diffusion resistance and diffusion capacitance if the diode is forward biased at (a) \( V_a = 0.970 \, \text{V} \) and (b) \( V_a = 1.045 \, \text{V} \).

**TYU 8.6**  
A silicon pn junction diode at \( T = 300 \, \text{K} \) has the same parameters as those described in Ex 8.7. The neutral n-region and neutral p-region lengths are 0.01 cm. Estimate the series resistance of the diode (neglect ohmic contacts).

----

## 8.4 | CHARGE STORAGE AND DIODE TRANSIENTS

The pn junction is typically used as an electrical switch. In forward bias, referred to as the on state, a relatively large current can be produced by a small applied voltage; in reverse bias, referred to as the off state, only a very small current will exist. Of primary interest in circuit applications is the speed of the pn junction diode in switching states. We qualitatively discuss the transients that occur and the charge storage effects. We simply state the equations that describe the switching times without any mathematical derivations.

### 8.4 Charge Storage and Diode Transients

!Figure 8.24  
**Figure 8.24** Simple circuit for switching a diode from forward to reverse bias.

#### 8.4.1 The Turn-off Transient

Suppose we want to switch a diode from the forward bias on state to the reverse-biased off state. Figure 8.24 shows a simple circuit that will switch the applied bias at \( t = 0 \). For \( t < 0 \), the forward-bias current is

\[
I = I_F = \frac{V_F - V_a}{R_F}
\]

(8.107)

The minority carrier concentrations in the device, for the applied forward voltage \( V_F \), are shown in Figure 8.25a. There is excess minority carrier charge stored in both the p and n regions of the diode. The excess minority carrier concentrations at the space charge edges are supported by the forward-bias junction voltage \( V_a \). When the voltage is switched from the forward- to the reverse-biased state, the excess minority carrier concentrations at the space charge edges can no longer be supported and they start to decrease, as shown in Figure 8.25b.

The collapse of the minority carrier concentrations at the edges of the space charge region leads to large concentration gradients and diffusion currents in the reverse-biased direction. If we assume, for the moment, that the voltage across the diode junction is small compared with \( V_R \), then the reverse-biased current is limited to approximately

\[
I = -I_R \approx -\frac{V_R}{R_R}
\]

(8.108)

The junction capacitances do not allow the junction voltage to change instantaneously. If the current \( I_R \) were larger than this value, there would be a forward-bias voltage across the junction, which would violate our assumption of a reverse-biased current.

# CHAPTER 8 The pn Junction Diode

!Figure 8.25

**Figure 8.25** (a) Steady-state forward-bias minority carrier concentrations; (b) minority carrier concentrations at various times during switching.

- **(a)**
  - \( n_p(x', t = 0) = n_{p0} \exp\left(\frac{eV_a}{kT}\right) \)
  - \( p_n(x, t = 0) = p_{n0} \exp\left(\frac{eV_a}{kT}\right) \)
  - Forward bias diffusion of electrons
  - Forward bias diffusion of holes

- **(b)**
  - Reverse bias diffusion of electrons
  - Reverse bias diffusion of holes
  - \( t = 0^- \)
  - \( t = 0^+ \)
  - \( t = t_1 \)
  - \( t_2 \)
  - \( t_3 \)
  - \( t_4 = t_s \)
  - \( t = \infty \)

!Figure 8.26

**Figure 8.26** Current characteristic versus time during diode switching.

- \( I_F \)
- Time
- \( -I_R \)
- \( t_1 \)
- \( t_2 = t_s \)
- \( 0.1 I_R \)

----

If the current \( I_R \) were smaller than this value, there would be a reverse-biased voltage across the junction, which means that the junction voltage would have changed instantaneously. Since the reverse current is limited to the value given by Equation (8.108), the reverse-biased density gradient is constant; thus, the minority carrier concentrations at the space charge edge decrease with time as shown in Figure 8.25b.

This reverse current \( I_R \) will be approximately constant for \( 0^+ \leq t \leq t_s \), where \( t_s \) is called the storage time. The storage time is the length of time required for the minority carrier concentrations at the space charge edge to reach the thermal-equilibrium values. After this time, the voltage across the junction will begin to change. The current characteristic is shown in Figure 8.26. The reverse current is the flow of the stored minority carrier charge, which is the difference between the minority carrier concentrations at \( t = 0^- \) and \( t = \infty \), as shown in Figure 8.25b.

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

# 8.5 The Tunnel Diode

The **tunnel diode** is a pn junction in which both the n and p regions are degenerately doped. As we discuss the operation of this device, we will find a region that exhibits a negative differential resistance. The tunnel diode was used in oscillator circuits in the past, but other types of solid-state devices are now used as high-frequency oscillators; thus, the tunnel diode is really only of academic interest. Nevertheless, this device does demonstrate the phenomenon of tunneling we discussed in Chapter 2.

Recall the degenerately doped semiconductors we discussed in Chapter 4: the Fermi level is in the conduction band of a degenerately doped n-type material and in the valence band of a degenerately doped p-type material. Then, even at \( T = 0 \, K \), electrons will exist in the conduction band of the n-type material, and holes (empty states) will exist in the p-type material.

Figure 8.27 shows the energy-band diagram of a pn junction in thermal equilibrium for the case when both the n and p regions are degenerately doped. The depletion region width decreases as the doping increases and may be on the order of approximately 100 Å for the case shown in Figure 8.27. The potential barrier at the junction can be approximated by a triangular potential barrier, as shown in Figure 8.28. This potential barrier is similar to the potential barrier used in Chapter 2 to illustrate the tunneling phenomenon. The barrier width is small and the electric field in the space charge region is quite large; thus, a finite probability exists that an electron may tunnel through the forbidden band from one side of the junction to the other.

We may qualitatively determine the current–voltage characteristics of the tunnel diode by considering the simplified energy-band diagrams in Figure 8.29.

!Energy-band diagram of a pn junction

**Figure 8.27** | Energy-band diagram of a pn junction in thermal equilibrium in which both the n and p regions are degenerately doped.

!Triangular potential barrier approximation

**Figure 8.28** | Triangular potential barrier approximation of the potential barrier in the tunnel diode.

# 8.5 The Tunnel Diode

!Energy-band diagrams and I-V characteristics

**Figure 8.29** | Simplified energy-band diagrams and \( I \text{-} V \) characteristics of the tunnel diode at (a) zero bias; (b) a slight forward bias; (c) a forward bias producing maximum tunneling current. *(continued)*

- **(a)** Zero bias: The energy-band diagram shows no applied voltage, and the corresponding \( I \text{-} V \) characteristic is at the origin.
- **(b)** Slight forward bias: The energy-band diagram indicates a small forward voltage, and the \( I \text{-} V \) curve begins to rise.
- **(c)** Forward bias producing maximum tunneling current: The energy-band diagram shows increased forward bias, and the \( I \text{-} V \) curve reaches a peak.

# Chapter 8 The pn Junction Diode

!Energy-band diagrams and I-V characteristics

**Figure 8.29** | (concluded) (d) A higher forward bias showing less tunneling current; (e) a forward bias for which the diffusion current dominates.

Figure 8.29a shows the energy-band diagram at zero bias, which produces zero current on the I–V diagram. If we assume, for simplicity, that we are near 0 K, then all energy states are filled below \( E_F \) on both sides of the junction.

Figure 8.29b shows the situation when a small forward-bias voltage is applied to the junction. Electrons in the conduction band of the n region are directly opposite to empty states in the valence band of the p region. There is a finite probability that some of these electrons will tunnel directly into the empty states, producing a forward-bias tunneling current as shown. With a slightly larger forward-bias voltage, as in Figure 8.29c, the maximum number of electrons in the n region will be opposite the maximum number of empty states in the p region; this will produce a maximum tunneling current.

As the forward-bias voltage continues to increase, the number of electrons on the n side directly opposite empty states on the p side decreases, as in Figure 8.29d, and the tunneling current will decrease. In Figure 8.29e, there are no electrons on the n side directly opposite to available empty states on the p side. For this forward-bias voltage, the tunneling current will be zero and the normal ideal diffusion current will exist in the device as shown in the I–V characteristics.

# 8.6 Summary

!Figure 8.30

**Figure 8.30** (a) Simplified energy-band diagram of a tunnel diode with a reverse-biased voltage; (b) I–V characteristic of a tunnel diode with a reverse-biased voltage.

The portion of the curve showing a decrease in current with an increase in voltage is the region of differential negative resistance. The range of voltage and current for this region is quite small; thus, any power generated from an oscillator using this negative resistance property would also be fairly small.

A simplified energy-band diagram of the tunnel diode with an applied reverse-biased voltage is shown in Figure 8.30a. Electrons in the valence band on the p side are directly opposite empty states in the conduction band on the n side, so electrons can now tunnel directly from the p region into the n region, resulting in a large reverse-biased tunneling current. This tunneling current will exist for any reverse-biased voltage. The reverse-biased current will increase monotonically and rapidly with reverse-biased voltage as shown in Figure 8.30b.

## 8.6.1 Summary

- When a forward-bias voltage is applied across a pn junction (p region positive with respect to the n region), the potential barrier is lowered so that holes from the p region and electrons from the n region can flow across the junction.
- The boundary conditions relating the minority carrier hole concentration in the n region at the space charge edge and the minority carrier electron concentration in the p region at the space charge edge were derived.
- The holes that are injected into the n region and the electrons that are injected into the p region now become excess minority carriers. The behavior of the excess minority carrier is described by the ambipolar transport equation developed and described in Chapter 6. Solving the ambipolar transport equation and using the boundary conditions, the steady-state minority carrier hole and electron concentrations in the n region and p region, respectively, were derived.
- Gradients exist in the minority carrier hole and electron concentrations so that minority carrier diffusion currents exist in the pn junction. These diffusion currents yield the ideal current–voltage relationship of the pn junction diode.

# Chapter 8: The pn Junction Diode

- Excess carriers are generated in the space charge region of a reverse-biased pn junction. These carriers are swept out by the electric field and create the reverse-biased generation current that is another component of the reverse-biased diode current. Excess carriers recombine in the space charge region of a forward-biased pn junction. This recombination process creates the forward-bias recombination current that is another component of the forward-bias diode current.

- The small-signal equivalent circuit of the pn junction diode was developed. The two parameters of interest are the diffusion resistance and the diffusion capacitance.

- When a pn junction is switched from forward bias to reverse bias, the stored excess minority carrier charge must be removed from the junction. The time required to remove this charge is called the storage time and is a limiting factor in the switching speed of a diode.

- The I–V characteristics of a tunnel diode were developed showing a region of negative differential resistance.

## Glossary of Important Terms

**carrier injection**  
The flow of carriers across the space charge region of a pn junction when a voltage is applied.

**diffusion capacitance**  
The capacitance of a forward-biased pn junction due to minority carrier storage effects.

**diffusion conductance**  
The ratio of a low-frequency, small-signal sinusoidal current to voltage in a forward-biased pn junction.

**diffusion resistance**  
The inverse of diffusion conductance.

**forward bias**  
The condition in which a positive voltage is applied to the p region with respect to the n region of a pn junction so that the potential barrier between the two regions is lowered below the thermal-equilibrium value.

**generation current**  
The reverse-biased pn junction current produced by the thermal generation of electron–hole pairs within the space charge region.

**high-level injection**  
The condition in which the excess carrier concentration becomes comparable to or greater than the majority carrier concentration.

**“long” diode**  
A pn junction diode in which both the neutral p and n regions are long compared with the respective minority carrier diffusion lengths.

**recombination current**  
The forward-bias pn junction current produced as a result of the flow of electrons and holes that recombine within the space charge region.

**reverse saturation current**  
The ideal reverse-biased current in a pn junction.

**“short” diode**  
A pn junction diode in which at least one of the neutral p or n regions is short compared to the respective minority carrier diffusion length.

**storage time**  
The time required for the excess minority carrier concentrations at the space charge edge to go from their steady-state values to zero when the diode is switched from forward to reverse bias.

## Checkpoint

After studying this chapter, the reader should have the ability to:

- Describe the mechanism of charge flow across the space charge region of a pn junction when a forward-bias voltage is applied.

## Problems

- State the boundary conditions for the minority carrier concentrations at the edge of the space charge region.
- Derive the expressions for the steady-state minority carrier concentrations in the pn junction.
- Derive the ideal current–voltage relationship for a pn junction diode.
- Describe the characteristics of a “short” diode.
- Describe generation and recombination currents in a pn junction.
- Define high-level injection and describe its effect on the diode I–V characteristics.
- Describe what is meant by diffusion resistance and diffusion capacitance.
- Describe the turn-off transient response in a pn junction.

## REVIEW QUESTIONS

1. Sketch the energy bands in a zero-biased, reverse-biased, and forward-biased pn junction.
2. Write the boundary conditions for the excess minority carriers in a pn junction (a) under forward bias and (b) under reverse bias.
3. Sketch the steady-state minority carrier concentrations in a forward-biased pn junction.
4. Explain the procedure that is used in deriving the ideal current–voltage relationship in a pn junction diode.
5. Sketch the electron and hole currents through a forward-biased pn junction diode. Are currents near the junction primarily due to drift or diffusion? What about currents far from the junction?
6. What is the temperature dependence of the ideal reverse-saturation current?
7. What is meant by a “short” diode?
8. Explain the physical mechanism of the (a) generation current and (b) recombination current.
9. Sketch the forward-bias I–V characteristics of a pn junction diode showing the effects of recombination and high-level injection.
10. (a) Explain the physical mechanism of diffusion capacitance. (b) What is diffusion resistance?
11. If a forward-biased pn junction is switched off, explain what happens to the stored minority carriers. In which direction is the current immediately after the diode is switched off?

## PROBLEMS

[Note: In the following problems, assume \( T = 300 \, K \) and the following parameters unless otherwise stated. For silicon pn junctions: \( D_n = 25 \, \text{cm}^2/\text{s}, D_p = 10 \, \text{cm}^2/\text{s}, \tau_{n0} = 5 \times 10^{-7} \, \text{s}, \tau_{p0} = 10^{-7} \, \text{s} \). For GaAs pn junctions: \( D_n = 205 \, \text{cm}^2/\text{s}, D_p = 9.8 \, \text{cm}^2/\text{s}, \tau_{n0} = 5 \times 10^{-8} \, \text{s}, \tau_{p0} = 10^{-8} \, \text{s} \).]

### Section 8.1 pn Junction Current

**8.1** (a) Consider an ideal pn junction diode at \( T = 300 \, K \) operating in the forward-bias region. Calculate the change in diode voltage that will cause a factor of 10 increase in current. (b) Repeat part (a) for a factor of 100 increase in current.

