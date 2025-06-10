# 6.3 Ambipolar Transport

Note that Equations (6.29) and (6.30) contain terms involving the total concentrations, \( p \) and \( n \), and terms involving only the excess concentrations, \( \delta p \) and \( \delta n \).

## 6.3.1 Ambipolar Transport

Originally, we assumed that the electric field in the current Equations (6.20) and (6.21) was an applied electric field. This electric field term appears in the time-dependent diffusion equations given by Equations (6.29) and (6.30). If a pulse of excess electrons and a pulse of excess holes are created at a particular point in a semiconductor with an applied electric field, the excess holes and electrons will tend to drift in opposite directions. However, because the electrons and holes are charged particles, any separation will induce an internal electric field between the two sets of particles. This internal electric field will create a force attracting the electrons and holes back toward each other. This effect is shown in Figure 6.5. The electric field term in Equations (6.29) and (6.30) is then composed of the externally applied field plus the induced internal field. This E-field may be written as

\[
E = E_{\text{app}} + E_{\text{int}}
\]

(6.31)

where \( E_{\text{app}} \) is the applied electric field and \( E_{\text{int}} \) is the induced internal electric field.

Since the internal E-field creates a force attracting the electrons and holes, this E-field will hold the pulses of excess electrons and excess holes together. The negatively charged electrons and positively charged holes then will drift or diffuse together with a single effective mobility or diffusion coefficient. This phenomenon is called **ambipolar diffusion** or **ambipolar transport**.

### 6.3.1 Derivation of the Ambipolar Transport Equation

The time-dependent diffusion Equations (6.29) and (6.30) describe the behavior of the excess carriers. However, a third equation is required to relate the excess electron and hole concentrations to the internal electric field. This relation is Poisson’s equation, which may be written as

\[
\nabla \cdot E_{\text{int}} = \frac{e(\delta p - \delta n)}{\varepsilon_s} = \frac{\partial E_{\text{int}}}{\partial x}
\]

(6.32)

where \( \varepsilon_s \) is the permittivity of the semiconductor material.

!Figure 6.5

**Figure 6.5** | The creation of an internal electric field as excess electrons and holes tend to separate.