# Chapter 15: Semiconductor Microwave and Power Devices

## Table 15.2 Characteristics of Two Power MOSFETs

| Parameter                  | 2N6757 | 2N6792 |
|----------------------------|--------|--------|
| \( V_{DS}(\text{max}) \) (V) | 150    | 400    |
| \( I_D(\text{max}) \) (at \( T = 25^\circ C \)) | 8      | 2      |
| \( P_D \) (W)              | 75     | 20     |

Resistance values are not necessarily negligible in power MOSFETs since small resistances and high currents can produce considerable power dissipation.

In the linear region of operation, we may write the channel resistance as:

\[
R_{CH} = \frac{L}{W \mu_n C_{ox} (V_{GS} - V_T)}
\]

We have noted in previous chapters that mobility decreases with increasing temperature. The threshold voltage varies only slightly with temperature so that, as current in a device increases and produces additional power dissipation, the temperature of the device increases, the carrier mobility decreases, and \( R_{CH} \) increases, which inherently limits the channel current. The resistances \( R_S \) and \( R_D \) are proportional to semiconductor resistivity and so are also inversely proportional to mobility and have the same temperature characteristics as \( R_{CH} \). Figure 15.23 shows a typical “on-resistance” characteristic as a function of drain current.

The increase in resistance with temperature provides stability for the power MOSFET. If the current in any particular cell begins to increase, the resulting temperature rise will increase the resistance, thus limiting the current. With this particular characteristic, the total current in a power MOSFET tends to be evenly distributed among the parallel cells, not concentrated in any single cell, a condition that can cause burnout.

!Figure 15.23

**Figure 15.23** Typical drain-to-source resistance versus drain current characteristics of a MOSFET.

- \( R_{DS(\text{on})} \) measured with current pulse of 2.0-μs duration, initial \( T_j = 25^\circ C \) (heating effect of 2.0-μs pulse is minimal)
- \( V_{GS} = 10 \, \text{V} \)
- \( V_{GS} = 20 \, \text{V} \)

The graph shows \( R_{DS(\text{on})} \) as a function of \( I_D \), drain current (A), with resistance (Ω) on the y-axis.