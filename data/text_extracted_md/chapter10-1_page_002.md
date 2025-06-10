# CHAPTER 10 Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

- Define and derive the expression for the threshold voltage, which is a basic parameter of the MOSFET.
- Discuss various physical structures of MOSFETs, including enhancement and depletion mode devices.
- Derive the ideal current–voltage relationship of the MOSFET.
- Develop the small-signal equivalent circuit of the MOSFET. This circuit is used to relate small-signal currents and voltages in analog circuits.
- Derive the frequency limiting factors of the MOSFET.

## 10.1 THE TWO-TERMINAL MOS STRUCTURE

The heart of the MOSFET is the MOS capacitor shown in Figure 10.1. The metal may be aluminum or some other type of metal, although in many cases, it is actually a high-conductivity polycrystalline silicon that has been deposited on the oxide; however, the term metal is usually still used. The parameter \( t_{\text{ox}} \) in the figure is the thickness of the oxide and \( \varepsilon_{\text{ox}} \) is the permittivity of the oxide.

### 10.1.1 Energy-Band Diagrams

The physics of the MOS structure can be more easily explained with the aid of the simple parallel-plate capacitor. Figure 10.2a shows a parallel-plate capacitor with the top plate at a negative voltage with respect to the bottom plate. An insulator material separates the two plates. With this bias, a negative charge exists on the top plate, a positive charge exists on the bottom plate, and an electric field is induced between the two plates as shown. The capacitance per unit area for this geometry is

\[
C' = \frac{\varepsilon}{d}
\]

(10.1)

where \( \varepsilon \) is the permittivity of the insulator and \( d \) is the distance between the two plates. The magnitude of the charge per unit area on either plate is

\[
Q' = C' V
\]

(10.2)

!Figure 10.1

**Figure 10.1** The basic MOS capacitor structure.