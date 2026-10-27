<div align="center">

# k-Nearest Neighbours

**Classification and regression with k-NN on three datasets**

<img src="https://img.shields.io/badge/Python-3-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python: 3" /> <img src="https://img.shields.io/badge/Pandas-data-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas: data" /> <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn: ML" /> <img src="https://img.shields.io/badge/Seaborn-charts-4C72B0?style=flat-square" alt="Seaborn: charts" />

</div>

## Overview

Supervised learning with **k-Nearest Neighbours**: preprocessing, training and prediction, evaluation with
cross-validation and comparison of real and predicted values.

| Notebook | Problem | Target | Result |
| --- | --- | --- | --- |
| [Heart](https://colab.research.google.com/github/Steven-Macas/SAA/blob/main/heart_SAA_stevenMacas_Nearest_Neighbour.ipynb) | Classification | Heart disease (yes / no) | 66.6 % accuracy |
| [Wine](https://colab.research.google.com/github/Steven-Macas/SAA/blob/main/vino_SAA_stevenMacas_Nearest_Neighbour.ipynb) | Classification | Wine quality | 42.4 % accuracy |
| [Paris housing](https://colab.research.google.com/github/Steven-Macas/SAA/blob/main/paris_SAA_stevenMacas_Nearest_Neighbour.ipynb) | Regression | House price | Predictions closely follow real prices |

Compared with the decision trees in [ArbolDecision](https://github.com/Steven-Macas/ArbolDecision), k-NN performs
worse on the heart (66.6 % vs 80.1 %) and wine (42.4 % vs 56.3 %) classification problems.

## Results

<table>
  <tr><td align="center" valign="top"><img src="images/heart-confusion-matrix.png" alt="Heart – confusion matrix" width="280" /><br /><sub>Heart – confusion matrix</sub></td><td align="center" valign="top"><img src="images/wine-confusion-matrix.png" alt="Wine – confusion matrix" width="280" /><br /><sub>Wine – confusion matrix</sub></td><td align="center" valign="top"><img src="images/paris-predictions.png" alt="Paris housing – real vs predicted" width="280" /><br /><sub>Paris housing – real vs predicted</sub></td></tr>
</table>

*The notebooks are written in Spanish.*

---

<sub>Made by <b>Steven Macas</b> — Data Engineer &amp; Data Analyst · <a href="https://gointy.com">gointy.com</a> · <a href="https://www.linkedin.com/in/steven-macas">LinkedIn</a> · <a href="https://github.com/Steven-Macas">GitHub</a></sub>
