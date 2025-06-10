# Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

As the collector current increases, the value of \( \alpha \) increases. As avalanche breakdown begins and \( I_C \) is small, particular values of \( M \) and \( V_{CE} \) are required to produce the condition of \( \alpha M = 1 \). As the collector current increases, \( \alpha \) increases; therefore, smaller values of \( M \) and \( V_{CE} \) are required to produce the avalanche breakdown condition. The snapback, or negative resistance, breakdown characteristic is then produced.

Only a fraction of the injected electrons from the forward-biased source-substrate junction are collected by the drain terminal. A more exact calculation of the snapback characteristic would necessarily involve taking into account this fraction; thus, the simple model would need to be modified. However, the above discussion qualitatively describes the snapback effect. The snapback effect can be minimized by using a heavily doped substrate that will prevent any significant voltage drop from being developed. A thin epitaxial p-type layer with the proper doping concentration to produce the required threshold voltage can be grown on a heavily doped substrate.

## Near Punch-Through Effects

Punch-through is the condition at which the drain-to-substrate space charge region extends completely across the channel region to the source-to-substrate space charge region. In this situation, the barrier between the source and drain is completely eliminated and a very large drain current would exist.

However, the drain current will begin to increase rapidly before the actual punch-through condition is reached. This characteristic is referred to as the near punch-through condition, also known as Drain-Induced Barrier Lowering (DIBL). Figure 11.25a shows the ideal energy-band diagram from source to drain for a long n-channel MOSFET for the case when \( V_{GS} < V_T \) and when the drain-to-source voltage is relatively small. The large potential barriers prevent significant current between the drain and source. Figure 11.25b shows the energy-band diagram when a relatively large drain voltage \( V_{DS2} \) is applied. The space charge region near the drain terminal is beginning to interact with the source space charge region and the potential barrier is being lowered. Since the current is an exponential function of barrier height, the current will increase very rapidly with drain voltage once this near punch-through condition is reached.

!Energy-band diagrams

**Figure 11.25** (a) Equipotential plot along the surface of a long-channel MOSFET.  
(b) Equipotential plot along the surface of a short-channel MOSFET before and after punch-through.