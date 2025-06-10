## 3.3.2 Additional Effective Mass Concepts

The curvature of the \( E \) versus \( k \) diagrams near the minimum of the conduction band energy is related to the effective mass of the electron. We may note from Figure 3.25 that the curvature of the conduction band at its minimum value for GaAs is larger than that of silicon, so the effective mass of an electron in the conduction band of GaAs will be smaller than that in silicon.

For the one-dimensional \( E \) versus \( k \) diagram, the effective mass was defined by Equation (3.41) as \( 1/m^* = 1/\hbar^2 \cdot d^2E/dk^2 \). A complication occurs in the effective mass concept in a real crystal. A three-dimensional crystal can be described by three \( k \) vectors. The curvature of the \( E \) versus \( k \) diagram at the conduction band minimum may not be the same in the three \( k \) directions. In later sections and chapters, the effective mass parameters used in calculations will be a kind of statistical average that is adequate for most device calculations.²

## 3.4 Density of States Function

As we have stated, we eventually wish to describe the current–voltage characteristics of semiconductor devices. Since current is due to the flow of charge, an important step in the process is to determine the number of electrons and holes in the semiconductor that will be available for conduction. The number of carriers that can contribute to the conduction process is a function of the number of available energy or quantum states since, by the Pauli exclusion principle, only one electron can occupy a given quantum state. When we discussed the splitting of energy levels into bands of allowed and forbidden energies, we indicated that the band of allowed energies was actually made up of discrete energy levels. We must determine the density of these allowed energy states as a function of energy in order to calculate the electron and hole concentrations.

### 3.4.1 Mathematical Derivation

To determine the density of allowed quantum states as a function of energy, we need to consider an appropriate mathematical model. Electrons are allowed to move relatively freely in the conduction band of a semiconductor but are confined to the crystal. As a first step, we will consider a free electron confined to a three-dimensional infinite potential well, where the potential well represents the crystal. The potential of the infinite potential well is defined as

\[
V(x, y, z) = 
\begin{cases} 
0 & \text{for } 0 < x < a \\ 
0 & \text{for } 0 < y < a \\ 
0 & \text{for } 0 < z < a 
\end{cases}
\]

\[
V(x, y, z) = \infty \quad \text{elsewhere}
\]

where the crystal is assumed to be a cube with length \( a \). Schrödinger’s wave equation in three dimensions can be solved by using the separation of variables technique.

----

²See Appendix F for further discussion of effective mass concepts.