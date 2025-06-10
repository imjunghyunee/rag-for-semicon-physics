```markdown
# CHAPTER 11 Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

The punch-through voltage is then found as

\[
V_{DS} = 5.77 - 0.874 = 4.9 \, \text{V}
\]

**Comment**  
As the two space charge regions approach punch-through, the abrupt junction approximation is no longer a good assumption. The drain current will begin to increase rapidly before the theoretical punch-through voltage is reached.

**EXERCISE PROBLEM**  
**Ex 11.5** Repeat Example 11.5 for a substrate doping concentration of \( N_a = 3 \times 10^{16} \, \text{cm}^{-3} \) and a channel length of \( L = 0.8 \, \mu\text{m} \).  
(\( \lambda \, \tau S L = 90 \, \text{A} \, \text{s/V} \))

For a doping of \( 10^{16} \, \text{cm}^{-3} \), the two space charge regions will begin to interact when the abrupt depletion layers are approximately 0.25 μm apart. The drain voltage at which this near punch-through condition, also known as drain-induced barrier lowering, occurs is significantly less than the ideal punch-through voltage such as calculated in Example 11.5 (see Problem 11.33).

## *11.4.2 The Lightly Doped Drain Transistor

The junction breakdown voltage is a function of the maximum electric field. As the channel length becomes smaller, the bias voltages may not be scaled down accordingly, so the junction electric fields become larger. As the electric field increases, near avalanche breakdown and near punch-through effects become more serious. In addition, as device geometries are scaled down, the parasitic bipolar device becomes more dominant and breakdown effects are enhanced.

One approach that reduces these breakdown effects is to alter the doping profile of the drain contact. The Lightly Doped Drain (LDD) design and doping profiles are shown in Figure 11.27a, the conventional MOSFET and doping profiles are shown in Figure 11.27b for comparison. By introducing the lightly doped region, the peak electric field in the space charge region is reduced and the breakdown effects are minimized. The peak electric field at the drain junction is a function of the semiconductor doping as well as the curvature of the n+ drain region. Figure 11.28 shows the physical geometries of a conventional n+ drain contact and an LDD structure superimposed on the same plot. The magnitude of the electric field at the oxide–semiconductor interface in the LDD structure is less than in the conventional structure. The electric field in the conventional device peaks approximately at the metallurgical junction and drops quickly to zero in the drain because no field can exist in the highly conductive n+ region. On the other hand, the electric field in the LDD device extends across the n-region before dropping to zero at the drain. This effect minimizes breakdown and the hot electron effects, which we discuss in Section 11.5.3.

Two disadvantages of the LDD device are an increase in both fabrication complexity and drain resistance. The added processing steps, however, produce a device with significant improvements in performance. The cross section of the LDD device
```