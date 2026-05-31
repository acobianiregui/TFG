# 📡 Bachelor's Thesis: Separation of Electromyographic Signals According to Their Muscle of Origin

This repository contains my Bachelor's Thesis in **Telecommunication Engineering**.  
The original thesis work was developed in Spanish, so some variable names, comments, notebooks or code sections may still appear in Spanish.

The project studies the **separation of weak electromyographic (EMG) signals mixed with stronger muscular interference**, with applications in active prostheses and reinnervated muscle scenarios such as **VDMT**, **RPNI** and **TMR**.

---

## 🎯 Objective
The main goal is to **recover weak EMG components from mixed recordings**, focusing especially on the preservation of their activation patterns.  
Several source separation approaches are explored, including **ICA, SOBI, constrained ICA and regression**, under controlled conditions of interference, noise and delay.

---

## 🛠️ Technologies used
- Languages: MATLAB, Python  
- Frameworks/Libraries: NumPy, SciPy, Matplotlib, Scikit-learn  
- Tools: MATLAB, Git, LaTeX  

---

## 📂 Repository structure
```
TFG/
├── codigo/ # Code for experiments and algorithms
│ ├── src/ # Core implementations (ICA, SOBI, preprocessing, metrics...)
│ ├── notebooks/ # Experiments and testing
│ ├── tools/ # Utility scripts
│ └── ...
│
├── latex/ # Thesis (LaTeX source)
│ ├── main.tex
│ ├── Secciones/
│ ├── anexos/
│ └── ...
│
├── figuras/ # Figures used in thesis
├── imagenes/ # Additional images/assets (related to Appendix B)
│
├── presentacion_seguimiento/ # Presentation slides
│
├── .github/ # CI workflows
├── README.md
└── LICENSE
```
---

## 📜 License

This project is distributed under the **MIT License**.  
You may use, modify and share it freely, provided that the author is credited.  

© 2025 Anton Cobian Iregui

For more details, see the [LICENSE](./LICENSE) file.

