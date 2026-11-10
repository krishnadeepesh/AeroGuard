# AeroGuard: Air Quality Category Prediction with a RAG-Grounded Health Advisory Assistant

## Overview
**AeroGuard** is an applied artificial intelligence system designed to monitor ambient air quality and translate complex atmospheric sensor telemetry into actionable, evidence-based public health guidance. 

Most public air monitoring platforms report raw numerical measurements and standard index numbers (such as an AQI value of 280), which leave everyday citizens unsure about practical risks—such as which specific pollutant is driving the danger, whether children or vulnerable individuals should remain indoors, or what protective gear is effective. AeroGuard bridges this gap by combining classical Machine Learning classification with Retrieval-Augmented Generation (RAG).

---

## How It Works

### 1. Machine Learning Triage
The predictive pipeline takes continuous measurements of eight atmospheric features:
* **Particulate Matter:** $\text{PM}_{2.5}$ and $\text{PM}_{10}$
* **Combustion & Toxic Gases:** $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, and $\text{O}_3$
* **Environmental Parameters:** Ambient Temperature and Relative Humidity

A trained **Random Forest Classifier** evaluates these features against standard Central Pollution Control Board (CPCB) National Air Quality Index (NAQI) criteria to predict the severity category:
* *Good*
* *Satisfactory*
* *Moderate*
* *Poor*
* *Very Poor*
* *Severe*

Simultaneously, the system calculates individual pollutant sub-indices to pinpoint the **dominant threat pollutant** responsible for current toxicity levels.

---

### 2. Evidence-Grounded Generative Advisory (RAG)
Rather than relying on unconstrained language models that risk hallucinations, AeroGuard connects the model's output directly to a local **Retrieval-Augmented Generation (RAG)** pipeline:
* **Knowledge Repository:** Curated guidelines based on World Health Organization (WHO) Global Air Quality Guidelines and Indian CPCB public health directives.
* **Semantic Retrieval:** Document passages are indexed into dense vector representations. When a category is predicted, the system retrieves relevant clinical and safety passages corresponding to that category and dominant pollutant.
* **Local Language Model:** A locally hosted Large Language Model (**Llama 3.2 via Ollama**) synthesizes the retrieved evidence into a structured public advisory with cited references.

---

### 3. Key Capabilities
* **Practical Health Directives:** Tailored recommendations for sensitive groups, including asthmatics, children, and the elderly.
* **Protective Measures:** Clear guidance on effective particulate filtration (e.g., certified N95 respirators versus cloth coverings) and indoor HEPA air purification.
* **Privacy & Local Execution:** The system runs completely on local hardware without sending telemetry or private queries to external cloud services.
* **Interactive Interface:** A web-based dashboard allowing users to adjust sensor readings, test real-world scenarios, inspect class confidence distributions, and generate grounded advisories in real time.
