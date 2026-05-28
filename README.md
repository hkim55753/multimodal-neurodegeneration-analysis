Computational Neuropsychiatric Biomarkers in Parkinson’s Disease
Overview

This project develops computational and statistical frameworks for analyzing neuropsychiatric, cognitive, and neuroimaging-derived biomarkers in Parkinson’s disease. The goal is to better understand disease progression by integrating motor, cognitive, and affective symptom domains with dopamine transporter imaging measures.

Using large-scale clinical datasets such as the Parkinson’s Progression Markers Initiative (PPMI), this work focuses on identifying quantitative relationships between dopaminergic dysfunction, cognitive decline, motor severity, and mood-related symptoms. The overarching aim is to build reproducible, multimodal pipelines for neuropsychiatric disease characterization and risk stratification.

Research Objectives
Develop computational models of neuropsychiatric and cognitive decline in Parkinson’s disease
Integrate multimodal clinical, behavioral, and imaging-derived biomarkers into unified analytic frameworks
Quantify relationships between dopaminergic dysfunction (DAT SPECT) and cognitive impairment
Examine interactions between motor severity, mood symptoms, and cognitive outcomes
Build scalable and reproducible pipelines for multimodal neurodegenerative disease analysis
Evaluate predictive models for neuropsychiatric risk stratification
Current Project Focus (PPMI Multimodal Biomarker Pipeline)

This project implements a reproducible computational pipeline using PPMI data, integrating:

Motor severity: MDS-UPDRS Part III
Cognitive function: Montreal Cognitive Assessment (MoCA)
Depressive symptoms: Geriatric Depression Scale (GDS)
Neuroimaging biomarkers: DAT SPECT striatal binding ratios

Derived features include:

Mean striatal dopamine transporter binding ratio
Cognitive impairment classification using standardized MoCA thresholds
Multimodal subject-level integration of imaging and clinical data
Key Analyses
Multimodal clinical-imaging correlation analysis
Association modeling between dopaminergic dysfunction and cognition
Logistic regression models for cognitive impairment prediction
Receiver operating characteristic (ROC) curve analysis for predictive performance
Stratification of patients using imaging-derived biomarkers
Visualization of multimodal relationships using correlation heatmaps and scatter plots
Data Sources

This project uses publicly available, de-identified datasets from:

Parkinson’s Progression Markers Initiative (PPMI)
DAT SPECT imaging-derived quantitative biomarker tables
Standardized clinical assessments across motor, cognitive, and neuropsychiatric domains

These datasets enable large-scale, reproducible investigation of neurodegenerative disease mechanisms.

Computational Tools
Python
pandas
NumPy
scikit-learn
matplotlib
seaborn
statsmodels
Clinical and Translational Significance

Parkinson’s disease is a progressive neurodegenerative disorder characterized by motor dysfunction, cognitive decline, and neuropsychiatric symptoms. These symptoms significantly impact quality of life and represent a major source of disability and healthcare burden.

This project contributes to computational approaches that may support:

Early identification of individuals at risk for cognitive impairment
Integration of imaging and clinical biomarkers for disease characterization
Development of scalable tools for multimodal neuropsychiatric analysis
Improved understanding of the relationship between dopaminergic dysfunction and cognition

By linking neuroimaging-derived biomarkers with clinical outcomes, this work supports translational research efforts in precision neurology and computational psychiatry.

Reproducibility

All analyses are implemented in Python using open-source scientific computing libraries. The pipeline is designed for reproducibility and extensibility, allowing validation across independent cohorts and future datasets.

Status

Active computational neuropsychiatric research project focused on multimodal biomarker development in Parkinson’s disease. The project is currently in the analysis and manuscript development stage, with ongoing expansion of predictive modeling and validation analyses.