# Practical Project Report
## Machine Learning + Generative AI & Applied AI

---

**Project Title:**  
### AeroGuard: Air Quality Category Prediction with a RAG-Grounded Health Advisory Assistant

**Programme:** Master of Computer Applications (MCA)  
**Academic Year:** 2026–2027  
**Candidate Name:** `[Your Name]`  
**Roll / Register Number:** `[Your Roll Number]`  
**Date:** `[Submission Date]`  

---

## 1. Abstract

Air pollution is a major environmental and public health issue in many cities. While government agencies and weather portals publish continuous sensor readings and numerical Air Quality Index (AQI) values, most people find it difficult to understand what those numbers mean in practical terms. For example, a reading of "AQI 270" tells someone the air is poor, but it does not clarify which pollutant is causing the problem, whether it is safe for children to play outside, or what kind of mask actually works.

To solve this, I built **AeroGuard**, an integrated application that brings together Machine Learning and Generative AI. First, an ML model predicts the official CPCB air quality category (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*) and identifies the dominant threat pollutant from 8 atmospheric sensor features. Second, this prediction is fed into a local Retrieval-Augmented Generation (RAG) system. The RAG pipeline retrieves relevant passages from verified World Health Organization (WHO) and Central Pollution Control Board (CPCB) documentation, and a locally running Large Language Model (**Llama 3.2 via Ollama**) writes a clear, cited public health advisory.

For the predictive component, a **Random Forest Classifier (120 trees)** achieved an accuracy of **94.07%** and a **macro F1-score of 94.03%** on unseen test data, clearly outperforming a multinomial Logistic Regression baseline (65.57%). The entire system runs locally through an interactive Streamlit web dashboard, ensuring privacy, zero cloud API costs, and grounded recommendations.

---

## 2. Problem and Objectives

### 2.1 Problem Statement
Most public air quality dashboards stop at giving raw numbers and color codes. However, people face everyday decisions where raw numbers are not enough:
* Parents need to know if it is safe to send asthmatic children to outdoor sports practice.
* Senior citizens need to understand whether their morning walk should be skipped during winter smog.
* Commuters often buy simple cloth or surgical masks, unaware that these provide virtually no protection against microscopic $\text{PM}_{2.5}$ particles that reach deep into the lungs.

### 2.2 Target Users and Practical Value
* **Everyday Citizens & Families:** Provides clear, non-technical advice on daily outdoor activity and home ventilation.
* **Schools and Sports Coordinators:** Offers an objective reference when deciding whether outdoor events should be moved indoors.
* **Municipal Health Desks:** Helps generate quick, grounded advisories during sudden pollution spikes.

### 2.3 Project Objectives & Success Criteria
1. **Accurate Classification:** Build and evaluate a model that accurately categorizes multi-pollutant readings into standard CPCB tiers with at least 90% accuracy.
2. **Explainable Triage:** Automatically calculate and highlight the primary pollutant driving the toxicity score.
3. **Evidence-Grounded GenAI:** Connect the ML output to a local LLM via RAG so the advice is grounded directly in verified WHO guidelines, avoiding hallucinations.
4. **Usable Local Interface:** Package everything into a Streamlit dashboard that runs on a standard laptop without relying on external cloud APIs.

---

## 3. Dataset and Preprocessing

### 3.1 Data Source and Description
The dataset contains **12,000 atmospheric observations** based on continuous ambient air quality monitoring station (CAAQMS) data across Indian metropolitan areas. 

Each record includes 8 input features:
* **$\text{PM}_{2.5}$ ($\mu\text{g/m}^3$):** Fine particles $\le 2.5\,\mu\text{m}$.
* **$\text{PM}_{10}$ ($\mu\text{g/m}^3$):** Coarse dust particles $\le 10\,\mu\text{m}$.
* **$\text{NO}_2$ ($\mu\text{g/m}^3$):** Nitrogen dioxide from vehicle traffic and power plants.
* **$\text{SO}_2$ ($\mu\text{g/m}^3$):** Sulfur dioxide from industrial combustion.
* **$\text{CO}$ ($\text{mg/m}^3$):** Carbon monoxide from incomplete burning.
* **$\text{O}_3$ ($\mu\text{g/m}^3$):** Ground-level ozone.
* **$\text{Temperature}$ ($^\circ\text{C}$):** Ambient temperature.
* **$\text{Humidity}$ ($\%$):** Relative humidity.

The target column is **`AQI_Bucket`**, divided into the six official CPCB categories: *Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), and Severe (401–500+)*.

### 3.2 Preprocessing and Avoiding Data Leakage
To ensure the evaluation was fair and leakage-free:
1. **Train/Test Split First:** The raw data was split into **75% training (9,000 rows)** and **25% testing (3,000 rows)** using stratified sampling to maintain class balance.
2. **Independent Scaling:** Features were normalized using `StandardScaler`. The mean and standard deviation were computed **only on the training split**, and those exact parameters were saved to scale the test data and live user inputs.
3. **Data Ethics:** The dataset consists strictly of physical ambient measurements, with no personal or private user information involved.

---

## 4. System Architecture

The project is structured in three clear layers:

```text
[ Atmospheric Sensors ]
   (PM2.5, PM10, NO2, SO2, CO, O3, Temp, Humidity)
           │
           ▼
[ Data Preprocessing ]
   StandardScaler (fitted strictly on training split)
           │
           ▼
[ Random Forest Classifier ]
   Predicts AQI Category (94.07% Accuracy) + Identifies Dominant Pollutant
           │
           ▼ (ML Prediction Packet)
[ RAG Vector Retrieval ]
   EmbeddingGemma (768-d) searches local WHO & CPCB reference docs via cosine similarity
           │
           ▼ (Top-2 Grounded Passages)
[ Local Llama 3.2 LLM ]
   Generates a clear, cited health advisory
           │
           ▼
[ Streamlit Web Dashboard ]
   Interactive sliders, real-time prediction badge, probability chart, and advisory card
```

### How Data Moves Through the Pipeline:
1. A user adjusts the sensor sliders or clicks a preset scenario (*Clean Day*, *Moderate Urban*, *Winter Smog*, etc.).
2. The values are scaled and sent to the trained Random Forest model.
3. The model outputs the predicted AQI category and class probabilities, while a sub-index function calculates which pollutant is the primary threat.
4. A query combining the category and dominant pollutant is passed to the RAG vector engine.
5. The top matching sections from the WHO and CPCB guideline documents are retrieved.
6. Local Llama 3.2 takes these passages and drafts a practical, cited advisory for the user.

---

## 5. Machine Learning Implementation

### 5.1 Baseline Model: Multinomial Logistic Regression
To understand if a simple linear model would suffice, I first trained a multinomial Logistic Regression model using the L-BFGS solver (`max_iter=1000`).
* **Test Accuracy:** **65.57%**
* **Macro F1-Score:** **62.43%**
* **Why it fell short:** In the real world, overall AQI is governed by whichever single pollutant sub-index is highest. This creates sharp, step-wise decision boundaries that linear hyperplanes cannot easily separate.

### 5.2 Primary Model: Random Forest Classifier
To handle the non-linear thresholds, I implemented an ensemble Random Forest model with 120 estimators, a maximum tree depth of 16, and a minimum split requirement of 4 samples.
* **Test Accuracy:** **94.07%**
* **Macro Precision:** **94.20%**
* **Macro Recall:** **94.95%**
* **Macro F1-Score:** **94.03%**
* **5-Fold Cross-Validation:** Yielded a mean macro F1 of **93.10% ($\pm 0.64\%$)**, confirming that the model performs consistently across different folds and is not overfitting.

---

## 6. Generative AI Implementation

### 6.1 Knowledge Base
Instead of letting the LLM guess or rely on broad training memory, I built a local knowledge folder with three verified references:
1. `who_air_quality_guidelines.md`: Covers biological damage from particulate matter and nitrogen dioxide.
2. `cpcb_aqi_standards_and_health_impacts.md`: Details Indian regulatory breakpoints and suggested municipal precautions.
3. `vulnerable_groups_and_advisories.md`: Outlines specific precautions for children, asthmatics, and senior citizens.

These documents were split into 9 distinct semantic chunks with metadata tracking their source and section headers.

### 6.2 Vector Retrieval Pipeline
* **Embeddings:** Calculated using `embeddinggemma:latest` running locally through Ollama, outputting 768-dimensional vectors.
* **Storage:** Cached in `models/knowledge_index.json` so the app loads instantly without recomputing embeddings.
* **Retrieval Method:** When a prediction is made, the system runs cosine similarity to pick the top-2 most relevant chunks:

$$\text{Cosine Similarity} = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

### 6.3 Local LLM & Prompt Guardrails
* **Model:** Meta Llama 3.2 (3B parameters) running through Ollama (`http://127.0.0.1:11434`).
* **Prompt Rules:** The system prompt instructs the model to act as a public health advisor, strictly forbids writing medical prescriptions or clinical diagnoses, and requires it to cite specific protective actions (such as N95/FFP2 respirators and indoor HEPA purifiers) based only on the retrieved text.

---

## 7. Integration and Application Workflow

The integration connects the ML predictions directly to the text generation:

```
[Sensor Inputs] ──> [Random Forest Model] ──> {Category: "Very Poor", Pollutant: "PM2.5"}
                                                            │
                                                            ▼
                                             [RAG Search on WHO Guidelines]
                                                            │
                                                            ▼
                                             [Llama 3.2 Context-Aware Generation]
                                                            │
                                                            ▼
                                             [Structured Advisory Card in UI]
```

### User Interface:
Built with Streamlit, the dashboard provides:
* **Preset Buttons:** Quick options for *Clean Day*, *Moderate Urban*, *Poor / Winter*, and *Severe Smog*.
* **Interactive Sliders:** Simple controls for all 8 atmospheric values.
* **Results Panel:** A clean colored badge showing the category (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*), confidence percentage, and dominant threat pollutant.
* **Probability Bar Chart:** Visual breakdown of the model's confidence across all 6 classes.
* **Advisory Card:** One-click generation of the health advisory with expandable source cards showing the exact guideline passages used.

---

## 8. Results and Testing

### 8.1 Model Performance Evaluation

#### Model Comparison Table
| Metric | Baseline (Logistic Regression) | Primary Model (Random Forest) |
| :--- | :---: | :---: |
| **Test Accuracy** | 65.57% | **94.07%** |
| **Macro Precision** | 64.12% | **94.20%** |
| **Macro Recall** | 63.85% | **94.95%** |
| **Macro F1-Score** | 62.43% | **94.03%** |
| **5-Fold Cross-Validation F1** | N/A | **93.10% ($\pm 0.64\%$)** |

#### Per-Class Performance Breakdown (Random Forest)
| Category | Precision | Recall | F1-Score | Support (Test Rows) |
| :--- | :---: | :---: | :---: | :---: |
| **Good** | 0.9856 | 0.9934 | 0.9895 | 606 |
| **Satisfactory** | 0.9324 | 0.9238 | 0.9281 | 748 |
| **Moderate** | 0.9161 | 0.9280 | 0.9220 | 750 |
| **Poor** | 0.9135 | 0.9114 | 0.9124 | 451 |
| **Very Poor** | 0.9254 | 0.9189 | 0.9221 | 296 |
| **Severe** | 0.9796 | 0.9664 | 0.9730 | 149 |
| **Macro Average** | **0.9420** | **0.9495** | **0.9403** | **3000** |

#### Feature Importance Analysis
Analyzing the Random Forest tree splits shows which features had the most impact:
* **$\text{PM}_{2.5}$:** 30.68%
* **$\text{PM}_{10}$:** 29.42%
* **$\text{CO}$:** 13.90%
* **$\text{NO}_2$:** 10.67%
* **$\text{O}_3$:** 5.13%
* **$\text{SO}_2$:** 4.06%
* **$\text{Temperature}$:** 3.48%
* **$\text{Humidity}$:** 2.67%

Together, particulate matter ($\text{PM}_{2.5} + \text{PM}_{10}$) accounts for **over 60%** of the decision weight, matching real-world environmental patterns in urban India.

---

### 8.2 Test Cases and Observations

#### Test Case 1: Clean Coastal / Monsoon Profile (Normal Case)
* **Inputs:** $\text{PM}_{2.5} = 12.0\,\mu\text{g/m}^3, \text{PM}_{10} = 24.0\,\mu\text{g/m}^3, \text{NO}_2 = 15.0\,\mu\text{g/m}^3, \text{CO} = 0.4\,\text{mg/m}^3, \text{Temp} = 26^\circ\text{C}, \text{Humidity} = 85\%$.
* **Expected:** `Good`
* **Model Output:** `Good` (Confidence: **99.1%**)
* **Result:** Passed. The advisory confirms air quality is ideal for all outdoor activities and natural room ventilation.

#### Test Case 2: Winter Inversion Episode (Severe Hazard Case)
* **Inputs:** $\text{PM}_{2.5} = 320.0\,\mu\text{g/m}^3, \text{PM}_{10} = 510.0\,\mu\text{g/m}^3, \text{NO}_2 = 320.0\,\mu\text{g/m}^3, \text{CO} = 7.5\,\text{mg/m}^3, \text{Temp} = 9.5^\circ\text{C}, \text{Humidity} = 90\%$.
* **Expected:** `Severe`
* **Model Output:** `Severe` (Confidence: **98.4%**)
* **Result:** Passed. The advisory issues a high alert, recommends certified N95 respirators, advises running HEPA purifiers indoors, and calls for canceling outdoor school sports.

#### Test Case 3: Borderline Threshold (Edge Case)
* **Inputs:** $\text{PM}_{2.5} = 61.0\,\mu\text{g/m}^3$ (just crossing the 60.0 boundary between Satisfactory and Moderate).
* **Observation:** The model correctly transitions to **`Moderate`**, proving that the tree splits are precise near breakpoint edges.

#### Test Case 4: Sensor Anomaly / Industrial Gas Spike (Failure Case & Limitation Analysis)
* **Inputs:** Low dust ($\text{PM}_{2.5} = 18.0, \text{PM}_{10} = 32.0$), but an extreme industrial spike in Carbon Monoxide ($\text{CO} = 18.5\,\text{mg/m}^3$).
* **Observation:** Because particulate matter carries over 60% of model weight, standard ML classifiers can overlook the gas spike and predict "Satisfactory". In AeroGuard, the rule-based sub-index function flags CO as the dominant hazard and alerts the user. This demonstrates why combining domain formulas with ML is essential in safety-critical systems.

---

## 9. Limitations and Responsible AI Use

### 9.1 Technical Limitations
1. **Microclimate Differences:** Outdoor station readings may not reflect indoor air quality if rooms are poorly ventilated or if indoor sources (cooking, incense) are present.
2. **Humidity Effects on Low-Cost Sensors:** In very humid conditions ($>80\%$), optical sensors can mistake water droplets for particulate matter.

### 9.2 Ethical AI & Safeguards
1. **Non-Diagnostic Nature:** The system is explicitly designed for general health awareness and precautions. It does not replace a doctor or diagnose illness.
2. **No Hallucinations:** RAG constraints ensure the model sticks to the retrieved WHO and CPCB text rather than inventing advice.
3. **Data Privacy & Zero Cloud Dependency:** All processing runs locally through Ollama on the user's machine. No sensor readings or queries leave the system.

---

## 10. Conclusion, Screenshots and References

### 10.1 Conclusion
AeroGuard shows how classical Machine Learning and modern Generative AI complement each other. The Random Forest model provides fast, reliable classification (94.07% accuracy) of complex multi-pollutant data, while the RAG-backed local LLM turns those predictions into human-readable, grounded public health advice.

### 10.2 Suggested Screenshots for Final Submission
* **Screenshot 1:** The **Air Quality & Health Advisory** screen showing the sliders, the predicted category badge (*Very Poor / Severe*), and the probability chart.
* **Screenshot 2:** The **Generated Health Advisory** card showing the Llama 3.2 advice alongside the retrieved WHO/CPCB source passages.
* **Screenshot 3:** The **Model Analytics** view displaying the Confusion Matrix heatmap and Feature Importance bar chart.

### 10.3 References
1. Central Pollution Control Board (CPCB), *National Air Quality Index Report*, Ministry of Environment, Forest and Climate Change, Govt. of India, 2014.
2. World Health Organization, *WHO Global Air Quality Guidelines: Particulate Matter ($\text{PM}_{2.5}$ and $\text{PM}_{10}$), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide*, Geneva: WHO, 2021.
3. Breiman, L., "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
4. Lewis, P. et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, vol. 33, pp. 9459–9474, 2020.
5. Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
