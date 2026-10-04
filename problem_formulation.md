# Problem Formulation: State-Preserving Intermittent TinyML

## 1. System Model & Assumptions
- **Intermittent Energy Harvesting:** The IoT node operates without a battery, harvesting energy into a small energy buffer (capacitor $C$). Power failures occur frequently when energy is exhausted ($V \le V_{min}$).
- **Non-Volatile Memory (NVM):** Persistent states are stored in FRAM/NVM, which incurs non-negligible energy and latency overheads for write operations.

## 2. Mathematical Formulation (Mục 3.1 & 7.2)
- Let $E_h(t)$ be the harvested energy rate from the environment (e.g., light traces from EnHANTs).
- Let $E_c$ be the energy consumed by executing neural network layers and state persistence.
- Objective: Maximize inference accuracy while ensuring forward progress and bounded overhead under dynamic energy constraints.
