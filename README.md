# Smart Fan Speed Controller
### Soft Computing and Applications (CSE320) &middot; Academic Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Report](https://img.shields.io/badge/Project%20Report-PDF%20(48%20Pages)-red.svg)](Project_Report.pdf)
[![Code style](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-brightgreen.svg)]()

A comprehensive Soft Computing project and real-time desktop simulator covering **Modules 01&ndash;03** of the **Soft Computing and Applications (CSE320)** curriculum at **Amity School of Engineering and Technology (ASET), Amity University, Noida**. 

The system implements, benchmarks, and demonstrates three distinct control paradigms to regulate an automated room cooling fan across environmental ambient temperature (18&ndash;42 &deg;C) and relative humidity (0&ndash;100%):
1. **Module 1 &middot; Crisp Logic:** Classical threshold control with hard decision boundaries.
2. **Module 2 &middot; Artificial Neural Networks:** A 2&ndash;6&ndash;1 Multilayer Perceptron (MLP) with $\tanh$ hidden units trained via analytical Backpropagation and Stochastic Gradient Descent (SGD).
3. **Module 3 &middot; Fuzzy Logic Systems:** A 9-rule Mamdani Fuzzy Inference System with partition-of-unity membership functions and centroid defuzzification.

---

### Student & Course Information
- **Student Name:** Anshumaan Singh
- **Enrollment Number:** A023167023097
- **Class / Section:** 7-DS-2(X)
- **Course Title:** Soft Computing and Applications (CSE320)
- **Course Instructor:** Dr. Smriti Sehgal
- **Institution:** Amity School of Engineering and Technology (ASET), Amity University, Noida
- **Academic Term:** November 2026
- **Complete Project Report:** [Download Project_Report.pdf (48 Pages)](Project_Report.pdf)

---

## Benchmark Results (Held-Out Test Grid)

All controllers are evaluated against a synthetic non-linear comfort policy with bilinear thermal-humidity interaction:
$$\text{Speed}_{\text{ref}} = 100 \times (0.05 + 0.63t + 0.12h + 0.16th)$$

Evaluation was conducted on **120 strictly held-out midpoint cases** disjoint from the **143 training points**:

| Controller Implementation | Module | Paradigm | Held-Out MAE | Max Error | Continuity / Physical Wear |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Crisp Thresholds** | Module 01 | Classical Logic | **13.79 pp** | 25.00 pp | Discontinuous (Step hunting, chattering) |
| **Mamdani Fuzzy Inference**| Module 03 | Soft Computing | **9.96 pp** | 17.42 pp | Smooth heuristic control; zero training data required |
| **MLP Neural Net ($\eta = 0.01$)** | Module 02 | Machine Learning | **0.98 pp** | 2.31 pp | Smooth continuous function approximation |
| **MLP Neural Net ($\eta = 0.08$)** | Module 02 | Machine Learning | **0.94 pp** | 2.14 pp | Fastest convergence and lowest generalization error |

*Note: pp = Percentage Points.*

---

## Key Technical Highlights

- **Zero Third-Party Dependencies:** Written in pure Python standard library (`math`, `random`, `dataclasses`, `tkinter`, `unittest`). Runs out-of-the-box on any Python 3.9+ installation without `numpy`, `scipy`, or `torch`.
- **Exact Analytical Backpropagation:** Implements exact chain-rule gradient calculus for the 2&ndash;6&ndash;1 MLP. Verified against two-sided numerical finite-difference gradients ($\epsilon = 10^{-6}$) to **$\ge 8$ decimal places**.
- **Fuzzy Partition of Unity:** Input triangular and shoulder membership functions satisfy $\sum \mu_i(x) = 1$ everywhere across the domain, guaranteeing zero dead-zones and unconditional defuzzification stability ($\min \sum \mu_{\text{agg}} = 13.0 > 0$).
- **Boundary Hunting Demonstration:** Demonstrates the severe real-world failure of crisp thresholds when moving between 29.9 &deg;C and 30.1 &deg;C (+25 pp jump) compared to the smooth response of fuzzy (+0.6 pp) and neural (+0.5 pp) models.

---

## Interactive Desktop Simulator

Launch the desktop laboratory GUI:

```bash
# Clone the repository
git clone https://github.com/Anshumaan-25/Smart-Fan-Speed-Controller.git
cd Smart-Fan-Speed-Controller

# Launch the simulator (on macOS)
./Launch-Fan-Lab.command

# Or directly with Python 3.9+
python3 fan_lab.py
```

### Features of the Simulator (`fan_lab.py`):
1. **Live Parametric Fan Animation:** Rotating 4-blade vector canvas animation whose angular velocity dynamically mirrors the active controller's speed.
2. **Temperature Sweep Analysis:** Real-time plots comparing Crisp steps, Fuzzy curve, Neural model, and the ground-truth target policy across 18&ndash;42 &deg;C.
3. **Fuzzy Rules Telemetry:** $3 \times 3$ matrix displaying live firing strengths $\alpha_{i,j}$ for all 9 rules alongside the aggregated membership polygon and centroid balance line.
4. **Logarithmic Training Loss Plot:** Multi-curve viewer tracking MSE loss descent across all 100 epochs for both learning rates.

---

## Automated Regression Tests

The test suite in `test_controllers.py` verifies boundary conditions, input rejection, fuzzy invariants, and analytical backpropagation gradients:

```bash
python3 -m unittest -v
```

Expected output:
```text
test_controllers_reject_invalid_conditions (test_controllers.ControllerTests) ... ok
test_crisp_thresholds_include_boundary (test_controllers.ControllerTests) ... ok
test_experiment_starts_equal_and_uses_disjoint_data (test_controllers.ControllerTests) ... ok
test_fuzzy_rule_aggregation_and_centroid (test_controllers.ControllerTests) ... ok
test_hidden_weight_update_matches_numerical_gradient (test_controllers.ControllerTests) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.059s

OK
```

---

## Repository Structure

```text
.
├── controllers.py            # Core algorithms, backprop engine, Mamdani FIS, experiment harness
├── fan_lab.py                # Desktop GUI simulator with vector canvas graphics and sweep plots
├── test_controllers.py       # Automated regression test suite and numerical gradient checks
├── Launch-Fan-Lab.command    # macOS double-clickable launcher script
├── Project_Report.pdf        # Complete 48-page formal academic submission report
├── README.md                 # Project documentation and quickstart
└── .gitignore                # Git exclusion rules
```

---

## References

1. L. A. Zadeh, "Fuzzy sets," *Information and Control*, vol. 8, no. 3, pp. 338–353, 1965.
2. E. H. Mamdani and S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," *Int. J. Man-Mach. Stud.*, vol. 7, no. 1, pp. 1–13, 1975.
3. D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, no. 6088, pp. 533–536, 1986.
4. S. Haykin, *Neural Networks and Learning Machines*, 3rd ed., Pearson Education, 2009.
5. T. J. Ross, *Fuzzy Logic with Engineering Applications*, 3rd ed., John Wiley & Sons, 2010.
