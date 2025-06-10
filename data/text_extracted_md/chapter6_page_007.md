```markdown
# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

## EXERCISE PROBLEM

**Ex 6.1** Using the parameters in Example 6.1, calculate the recombination rate of the excess carriers for (a) \( t = 0 \), (b) \( t = 1 \mu s \), (c) \( t = 4 \mu s \), and (d) \( t = 10 \mu s \).

\[ 
\text{[1.5 \times 10^{16} \, \text{cm}^{-3}]} \times e^{-t/\tau} \, (p) : i_{s} = u_{0} \, 0 \, 1 \, < \, 8 \, 1 \, (p) : i_{s} = u_{0} \, 0 \, 1 \, \times \, 8 \, 9 \, (q) : i_{s} = u_{0} \, r(0 \, 1) \, \text{suV}
\]

## 6.2 CHARACTERISTICS OF EXCESS CARRIERS

The generation and recombination rates of excess carriers are important parameters, but how the excess carriers behave with time and in space in the presence of electric fields and density gradients is of equal importance. As mentioned at the beginning of this chapter, the excess electrons and holes do not move independently of each other, but they diffuse and drift with the same effective diffusion coefficient and with the same effective mobility. This phenomenon is called ambipolar transport. The question that must be answered is what is the effective diffusion coefficient and what is the effective mobility that characterizes the behavior of these excess carriers? To answer these questions, we must develop the continuity equations for the carriers and then develop the ambipolar transport equations.

The final results show that, for an extrinsic semiconductor under low injection (this concept will be defined in the analysis), the effective diffusion coefficient and mobility parameters are those of the minority carrier. This result is thoroughly developed in the following derivations. As will be seen in the following chapters, the behavior of the excess carriers has a profound impact on the characteristics of semiconductor devices.

### 6.2.1 Continuity Equations

The continuity equations for electrons and holes are developed in this section. Figure 6.4 shows a differential volume element in which a one-dimensional hole-particle flux is entering the differential element at \( x \) and is leaving the element at \( x + dx \). The parameter \( F_{px}^{+} \) is the hole-particle flux, or flow, and has units of number of holes/cm\(^2\)-s. For the \( x \) component of the particle current density shown, we may write

\[
F_{px}^{+}(x + dx) = F_{px}^{+}(x) + \frac{\partial F_{px}^{+}}{\partial x} \cdot dx
\]

\[
(6.15)
\]

!Figure 6.4

**Figure 6.4** | Differential volume showing \( x \) component of the hole-particle flux.
```