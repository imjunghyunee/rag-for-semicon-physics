# 5.1 CARRIER DRIFT

An electric field applied to a semiconductor will produce a force on electrons and holes so that they will experience a net acceleration and net movement, provided there are available energy states in the conduction and valence bands. This net movement of charge due to an electric field is called **drift**. The net drift of charge gives rise to a **drift current**.

## 5.1.1 Drift Current Density

If we have a **positive volume charge density** \( \rho \) moving at an **average drift velocity** \( v_d \), the **drift current density** is given by

\[
J_{dr} = \rho v_d
\]

(5.1a)

In terms of units, we have

\[
J_{dr} = \left( \frac{\text{Coul}}{\text{cm}^3} \right) \left( \frac{\text{cm}}{\text{s}} \right) = \frac{\text{Coul}}{\text{cm}^2 \cdot \text{s}} = \frac{\text{A}}{\text{cm}^2}
\]

(5.1b)

If the **volume charge density is due to positively charged holes**, then

\[
J_{pdr} = (eP)v_{dp}
\]

(5.2)

where \( J_{pdr} \) is the drift current density due to holes and \( v_{dp} \) is the average drift velocity of the holes.

The equation of motion of a positively charged hole in the presence of an electric field is

\[
F = m_p^* a = eE
\]

(5.3)

where \( e \) is the magnitude of the electronic charge, \( a \) is the acceleration, \( E \) is the electric field, and \( m_p^* \) is the conductivity effective mass of the hole. If the electric field is constant, then we expect the velocity to increase linearly with time. However, charged particles in a semiconductor are involved in collisions with ionized impurity atoms and with thermally vibrating lattice atoms. These collisions, or scattering events, alter the velocity characteristics of the particle.

As the hole accelerates in a crystal due to the electric field, the velocity increases. When the charged particle collides with an atom in the crystal, for example, the particle loses most, or all, of its energy. The particle will again begin to accelerate and gain energy until it is again involved in a scattering process. This continues over and over again. Throughout this process, the particle will gain an average drift velocity which, for low electric fields, is directly proportional to the electric field. We may then write

\[
v_{dp} = \mu_p E
\]

(5.4)

where \( \mu_p \) is the proportionality factor and is called the **hole mobility**. The mobility is an important parameter of the semiconductor since it describes how well a particle can move through the material.

*The conductivity effective mass is used when carriers are in motion. See Appendix F for further discussion of effective mass concepts.*