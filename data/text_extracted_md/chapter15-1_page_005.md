# Chapter 15: Semiconductor Microwave and Power Devices

!Figure 15.5

**Figure 15.5** (a) A simplified two-terminal GaAs device. (b) Electron concentration versus distance showing a space charge formation. (c) Electric field versus distance.

In discussing excess carrier behavior in Chapter 6, we found the time behavior of a net charge density in a semiconductor to be given by

\[
\delta Q(t) = \delta Q(0)e^{-t/\tau_d}
\]

(15.3)

where \(\tau_d\) is the dielectric relaxation time constant and is on the order of a picosecond. Normally, a small space charge region would be quickly neutralized. The dielectric relaxation time constant is given by \(\tau_d = \varepsilon/\sigma\), where \(\sigma\) is the semiconductor conductivity. If the GaAs is biased in the negative mobility region, then the conductivity is negative and the exponent in Equation (15.3) becomes positive, so the space charge region, now called a domain, can actually build up as it drifts toward the anode. As the domain grows (Figure 15.6a), the electric field in this region increases which means that the electric field in the remaining material decreases. The E field in the material outside of the domain can drop below the critical value, as indicated in Figure 15.6b, while the E field within the domain remains above the critical value. For this reason, only one domain will normally be established in the material at any given time.

As the domain reaches the anode, a current pulse is induced in the external circuit. After the domain reaches the anode, another domain may form near the cathode and the process repeats itself. Thus, a series of current pulses may be generated as shown in Figure 15.7. The time between current pulses is the time for the domain to drift through the device. The oscillation frequency is given by

\[
f = 1/\tau = v_d/L
\]

(15.4)

where \(v_d\) is the average drift velocity and \(L\) is the length of the drift region.