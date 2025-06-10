```markdown
# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

It is extremely important to note that the transport and recombination parameters in Equations (6.55) and (6.56) are those of the minority carrier. **Equations (6.55) and (6.56) describe the drift, diffusion, and recombination of excess minority carriers as a function of spatial coordinates and as a function of time.** Recall that we had imposed the condition of charge neutrality; the excess minority carrier concentration is equal to the excess majority carrier concentration. The excess majority carriers, then, diffuse and drift with the excess minority carriers; thus, the behavior of the excess majority carrier is determined by the minority carrier parameters. This ambipolar phenomenon is extremely important in semiconductor physics, and is the basis for describing the characteristics and behavior of semiconductor devices.

## 6.3.3 Applications of the Ambipolar Transport Equation

We solve the ambipolar transport equation for several problems. These examples help illustrate the behavior of excess carriers in a semiconductor material, and the results are used later in the discussion of the pn junction and the other semiconductor devices.

The following examples use several common simplifications in the solution of the ambipolar transport equation. Table 6.2 summarizes these simplifications and their effects.

### EXAMPLE 6.2

**Objective:** Determine the time behavior of excess carriers as a semiconductor returns to thermal equilibrium.

Consider an infinitely large, homogeneous n-type semiconductor with zero applied electric field. Assume that at time \( t = 0 \), a uniform concentration of excess carriers exists in the crystal, but assume that \( g' = 0 \) for \( t > 0 \). If we assume that the concentration of excess carriers is much smaller than the thermal-equilibrium electron concentration, then the low-injection condition applies. Calculate the excess carrier concentration as a function of time for \( t \geq 0 \).

### Table 6.2 | Common ambipolar transport equation simplifications

| Specification                                      | Effect                                                                 |
|----------------------------------------------------|------------------------------------------------------------------------|
| Steady state                                       | \(\frac{\partial (\delta n)}{\partial t} = 0, \quad \frac{\partial (\delta p)}{\partial t} = 0\) |
| Uniform distribution of excess carriers            | \(D_n \frac{\partial^2 (\delta n)}{\partial x^2} = 0, \quad D_p \frac{\partial^2 (\delta n)}{\partial x^2} = 0\) |
| (uniform generation rate)                          |                                                                        |
| Zero electric field                                | \(E \frac{\partial (\delta n)}{\partial x} = 0, \quad E \frac{\partial (\delta p)}{\partial x} = 0\) |
| No excess carrier generation                       | \(g' = 0\)                                                             |
| No excess carrier recombination                    | \(\frac{\delta n}{\tau_{n0}} = 0, \quad \frac{\delta p}{\tau_{p0}} = 0\) |
| (infinite lifetime)                                |                                                                        |
```