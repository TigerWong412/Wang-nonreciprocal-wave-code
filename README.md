# 🌊 Code for: Non-Reciprocal Active Pair Transport in a Traveling Wave

This repository contains 2 Python scripts. One Python script "non_reciprocal_1D.py" is used to simulate the dynamics of a non-reciprocal active particle pair subject to a one-dimensional traveling square wave. The other, "non_reciprocal_2D.py", simulates the dynamics of a population of active particles, or a mixture of active and passive particles, in 2D traveling square waves. Each code has a Readme.txt file.

These two codes model the overdamped Brownian dynamics (BD) of two particles with **non-reciprocal pairwise forces** (attraction/repulsion determined by the local phase of the external wave) and a **hard-core contact constraint**. These codes are used for the research work in the research article "Non-reciprocal Colloidal Transport Guided by Traveling Stimulus Waves", submitted by Peng, Khan et al.

Read the specific readme.txt file of each code for more detailed information.

Email Prof. Wei Wang at weiwangsz@hit.edu.cn for questions.

---

## 📄 Citation Information

If you use this code in your work, please cite the associated publication (to be published):

> [Author Names]. (Year). [Full Title of Research Article]. *[Journal Name], [Volume](Issue), [Page Numbers].* DOI: [Insert DOI Here]

---

## ⚙️ Requirements and Dependencies

The simulation is implemented in **Python 3.9+**. The following external libraries (and possibily more) are required:

* **numpy** (for numerical operations)
* **matplotlib** (for plotting)
* **imageio** (specifically `imageio.v2` for video generation)
* **os** and **shutil** (standard Python libraries for file/directory management)

You can install the dependencies using pip:

```bash
pip install numpy matplotlib imageio
