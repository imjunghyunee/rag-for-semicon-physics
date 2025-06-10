# Chapter 4: The Semiconductor in Equilibrium

very simple. It is one of the fundamental principles of semiconductors in thermal equilibrium. The significance of this relation will become more apparent in the chapters that follow. It is important to keep in mind that Equation (4.43) was derived using the Boltzmann approximation. If the Boltzmann approximation is not valid, then likewise, Equation (4.43) is not valid.

An extrinsic semiconductor in thermal equilibrium does not, strictly speaking, contain an intrinsic carrier concentration, although some thermally generated carriers are present. The intrinsic electron and hole carrier concentrations are modified by the donor or acceptor impurities. However, we may think of the intrinsic concentration \( n_i \) in Equation (4.43) simply as a parameter of the semiconductor material.

## 4.3.3 The Fermi–Dirac Integral

In the derivation of the Equations (4.11) and (4.19) for the thermal equilibrium electron and hole concentrations, we assumed that the Boltzmann approximation was valid. If the Boltzmann approximation does not hold, the thermal equilibrium electron concentration is written from Equation (4.43) as

\[
n_0 = \frac{4 \pi}{h^3} (2m_n^*)^{3/2} \int_{E_c}^{\infty} \frac{(E - E_c)^{1/2} \, dE}{1 + \exp \left( \frac{E - E_f}{kT} \right)}
\]

(4.44)

If we again make a change of variable and let

\[
\eta = \frac{E - E_c}{kT}
\]

(4.45a)

and also define

\[
\eta_f = \frac{E_f - E_c}{kT}
\]

(4.45b)

then we can rewrite Equation (4.44) as

\[
n_0 = 4 \pi \left( \frac{2m_n^* kT}{h^2} \right)^{3/2} \int_0^{\infty} \frac{\eta^{1/2} \, d\eta}{1 + \exp (\eta - \eta_f)}
\]

(4.46)

The integral is defined as

\[
F_{1/2}(\eta_f) = \int_0^{\infty} \frac{\eta^{1/2} \, d\eta}{1 + \exp (\eta - \eta_f)}
\]

(4.47)

This function, called the Fermi–Dirac integral, is a tabulated function of the variable \( \eta_f \). Figure 4.10 is a plot of the Fermi–Dirac integral. Note that if \( \eta_f > 0 \), then \( E_f > E_c \); thus, the Fermi energy is actually in the conduction band.

----

### Example 4.6

**Objective:** Calculate the electron concentration using the Fermi–Dirac integral.

Let \( \eta_f = 2 \) so that the Fermi energy is above the conduction band by approximately 52 meV at \( T = 300 \, \text{K} \).

**Solution**

Equation (4.46) can be written as

\[
n_0 = \frac{2}{\sqrt{\pi}} N_c F_{1/2} (\eta_f)
\]