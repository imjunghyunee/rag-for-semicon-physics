### 4.1 Charge Carriers in Semiconductors

Figure 4.1b is an expanded view of the plot in Figure 4.1a showing \( g_c(E) \) and \( f(E) \) above the conduction-band energy \( E_c \). The product of \( g_c(E) \) and \( f(E) \) is the distribution of electrons \( n(E) \) in the conduction band given by Equation (4.1). This product is plotted in Figure 4.1a. Figure 4.1c, which is an expanded view of the plot in Figure 4.1a shows \([1 - f_f(E)]\) and \( g_v(E) \) below the valence-band energy \( E_v \). The product of \( g_v(E) \) and \([1 - f_f(E)]\) is the distribution of holes \( p(E) \) in the valence band given by Equation (4.2). This product is also plotted in Figure 4.1a. The areas under these curves are then the total density of electrons in the conduction band and the total density of holes in the valence band. From this we see that if \( g_c(E) \) and \( g_v(E) \) are symmetrical, the Fermi energy must be at the midgap energy in order to obtain equal electron and hole concentrations. If the effective masses of the electron and hole are not exactly equal, then the effective density of states functions \( g_c(E) \) and \( g_v(E) \) will not be exactly symmetrical about the midgap energy. The Fermi level for the intrinsic semiconductor will then shift slightly from the midgap energy in order to obtain equal electron and hole concentrations.

#### 4.1.2 The \( n_0 \) and \( p_0 \) Equations

We have argued that the Fermi energy for an intrinsic semiconductor is near midgap. In deriving the equations for the thermal-equilibrium concentration of electrons \( n_0 \) and the thermal-equilibrium concentration of holes \( p_0 \), we will not be quite so restrictive. We will see later that, in particular situations, the Fermi energy can deviate from this midgap energy. We will assume initially, however, that the Fermi level remains within the bandgap energy.

**Thermal-Equilibrium Electron Concentration** The equation for the thermal-equilibrium concentration of electrons may be found by integrating Equation (4.1) over the conduction band energy, or

\[
n_0 = \int_{E_c}^{\infty} g_c(E)f_f(E) \, dE
\]

(4.3)

The lower limit of integration is \( E_c \) and the upper limit of integration should be the top of the allowed conduction band energy. However, since the Fermi probability function rapidly approaches zero with increasing energy as indicated in Figure 4.1a, we can take the upper limit of integration to be infinity.

We are assuming that the Fermi energy is within the forbidden-energy bandgap. For electrons in the conduction band, we have \( E > E_c \). If \((E - E_f) \gg kT\), then \((E - E_c) \gg kT\), so that the Fermi probability function reduces to the Boltzmann approximation, which is

\[
f_f(E) = \frac{1}{1 + \exp\left(\frac{E - E_f}{kT}\right)} \approx \exp\left[-\frac{(E - E_f)}{kT}\right]
\]

(4.4)

*The Maxwell–Boltzmann and Fermi–Dirac distribution functions are within 5 percent of each other when \( E - E_f > 3kT \) (see Figure 3.35). The \(\gg\) notation is then somewhat misleading to indicate when the Boltzmann approximation is valid, although it is commonly used.*