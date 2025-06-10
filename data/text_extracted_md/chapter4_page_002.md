# 4.1 Charge Carriers in Semiconductors

Current is the rate at which charge flows. In a semiconductor, two types of charge carriers, the electron and the hole, can contribute to a current. Since the current in a semiconductor is determined largely by the number of electrons in the conduction band and the number of holes in the valence band, an important characteristic of the semiconductor is the density of these charge carriers. The density of electrons and holes is related to the density of states function and the Fermi distribution function, both of which we have considered. A qualitative discussion of these relationships will be followed by a more rigorous mathematical derivation of the thermal-equilibrium concentration of electrons and holes.

## 4.1.1 Equilibrium Distribution of Electrons and Holes

The distribution (with respect to energy) of electrons in the conduction band is given by the density of allowed quantum states times the probability that a state is occupied by an electron. This statement is written in equation form as

\[
n(E) = g_c(E)f_r(E)
\]

(4.1)

where \( f_r(E) \) is the Fermi–Dirac probability function and \( g_c(E) \) is the density of quantum states in the conduction band. The total electron concentration per unit volume in the conduction band is then found by integrating Equation (4.1) over the entire conduction-band energy.

Similarly, the distribution (with respect to energy) of holes in the valence band is the density of allowed quantum states in the valence band multiplied by the probability that a state is not occupied by an electron. We may express this as

\[
p(E) = g_v(E)[1 - f_r(E)]
\]

(4.2)

The total hole concentration per unit volume is found by integrating this function over the entire valence-band energy.

To find the thermal-equilibrium electron and hole concentrations, we need to determine the position of the Fermi energy \( E_f \) with respect to the bottom of the conduction-band energy \( E_c \) and the top of the valence-band energy \( E_v \). To address this question, we will initially consider an intrinsic semiconductor. An ideal intrinsic semiconductor is a pure semiconductor with no impurity atoms and no lattice defects in the crystal (e.g., pure silicon). We have argued in the previous chapter that, for an intrinsic semiconductor at \( T = 0 \, K \), all energy states in the valence band are filled with electrons and all energy states in the conduction band are empty of electrons. The Fermi energy must, therefore, be somewhere between \( E_v \) and \( E_c \). (The Fermi energy does not need to correspond to an allowed energy.)

As the temperature begins to increase above 0 K, the valence electrons will gain thermal energy. A few electrons in the valence band may gain sufficient energy to jump to the conduction band. As an electron jumps from the valence band to the conduction band, an empty state, or hole, is created in the valence band. In an intrinsic