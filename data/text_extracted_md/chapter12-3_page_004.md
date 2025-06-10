# Chapter 12: The Bipolar Transistor

!Figure 12.34

**Figure 12.34** (a) Open-emitter configuration with saturation current \( I_{CBO} \); (b) Open-base configuration with saturation current \( I_{CEO} \).

The condition also makes the B–C junction reverse biased. The current in the transistor for this bias configuration is denoted as \( I_{CBO} \).

The current \( I_{CBO} \) shown in Figure 12.34b is the normal reverse-biased B–C junction current. Part of this current is due to the flow of minority carrier holes from the collector across the B–C space charge region into the base. The flow of holes into the base makes the base positive with respect to the emitter, and the B–E junction becomes forward biased. The forward-biased B–E junction produces the current \( I_{EBO} \) due primarily to the injection of electrons from the emitter into the base. The injected electrons diffuse across the base toward the B–C junction. These electrons are subject to all of the recombination processes in the bipolar transistor. When the electrons reach the B–C junction, this current component is \( \alpha I_{CBO} \) where \( \alpha \) is the common-base current gain. We therefore have

\[
I_{CEO} = \alpha I_{CBO} + I_{CBO}
\]

(12.55a)

or

\[
I_{CEO} = \frac{I_{CBO}}{1 - \alpha} \approx \beta I_{CBO}
\]

(12.55b)

where \( \beta \) is the common-emitter current gain. The reverse-biased junction current \( I_{CBO} \) is multiplied by the current gain \( \beta \) when the transistor is biased in the open-base configuration.

When the transistor is biased in the open-emitter configuration as in Figure 12.34a, the current \( I_{CBO} \) at breakdown becomes \( I_{CBO} \rightarrow MI_{CBO} \), where \( M \) is the multiplication factor. An empirical approximation for the multiplication factor is usually written as

\[
M = \frac{1}{1 - (V_{CB}/BV_{CBO})^n}
\]

(12.56)

where \( n \) is an empirical constant, usually between 3 and 6, and \( BV_{CBO} \) is the B–C breakdown voltage with the emitter left open.

When the transistor is biased with the base open circuited as shown in Figure 12.34b, the currents in the B–C junction at breakdown are multiplied, so that

\[
I_{CEO} = M(\alpha I_{CBO} + I_{CBO})
\]

(12.57)

Solving for \( I_{CEO} \), we obtain

\[
I_{CEO} = \frac{MI_{CBO}}{1 - \alpha M}
\]

(12.58)