# 🌊 Code for: Non-Reciprocal Active Pair Transport in a Traveling Wave

This repository contains 2 Python scripts. One Python script is used to simulate the dynamics of a non-reciprocal active particle pair subject to a one-dimensional traveling square wave, as discussed in the associated research article. The other simulates the dynamics of a population of active particles, or a mixture of active and passive particles, in 2D traveling square waves.

The script models the overdamped Brownian dynamics (BD) of two particles with **non-reciprocal pairwise forces** (attraction/repulsion determined by the local phase of the external wave) and a **hard-core contact constraint**.

---

## 📄 Citation Information

If you use this code in your work, please cite the associated publication:

> [Author Names]. (Year). [Full Title of Research Article]. *[Journal Name], [Volume](Issue), [Page Numbers].* DOI: [Insert DOI Here]

---

## ⚙️ Requirements and Dependencies

The simulation is implemented in **Python 3.9+**. The following external libraries are required:

* **numpy** (for numerical operations)
* **matplotlib** (for plotting)
* **imageio** (specifically `imageio.v2` for video generation)
* **os** and **shutil** (standard Python libraries for file/directory management)

You can install the dependencies using pip:

```bash
pip install numpy matplotlib imageio
