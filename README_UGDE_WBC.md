TITLE
A Federated Deep Learning Framework for Leukemia Diagnosis Using Multi-Level Explainability

Description
This project the implementation of a Federated Deep Learning Framework for Leukemia Diagnosis Using Multi-Level Explainability (FLLD).
The proposed framework combines Federated Learning, Deep Learning, and Explainable Artificial Intelligence for the classification of Acute Lymphoblastic Leukemia from peripheral blood smear images.
The framework uses a pre-trained ResNet-34 model for four-class leukemia classification and enables collaborative model training across distributed clients without directly sharing raw medical images. Federated model updates are aggregated using Federated Averaging (FedAvg).
To improve transparency and interpretability, the trained global model is analyzed using three complementary XAI techniques:
Grad-CAM, LIME, SHAP
The framework is designed to support privacy-preserving collaborative learning while providing visual and feature-level explanations of model predictions.
Dataset Information
The experiments use a publicly available ALL blood-smear image dataset obtained from the Multi Cancer Dataset repository on Kaggle.
The original dataset contains images associated with ALL and benign/hematogone samples. For this study, the images were organized into four classification categories:
ClassDescription
	/All_Benign	
	/All_Early	
	/All_Pre 
	/All_Pro
For the proposed experiments, a balanced dataset of 20,000 images was prepared, with 5,000 images per class.
Code Overview
The pipeline integrates:
1. Data Preprocessing & Preparation
   Organizes the leukemia blood-smear images into four classes:All_Benign, All_Early, All_Pr`, and All_Pro.
   Resizes images to 224 × 224 × 3 and applies normalization and image quality enhancement.
   Uses a balanced dataset of 20,000 images, with 5,000 images per class.
   Splits the dataset into 70% training, 15% validation, and 15% testing.
2. Federated Model Training
   * Uses a pre-trained ResNet-34 model for four-class leukemia classification.
   * Simulates three distributed clients representing hospitals or diagnostic centers.
   * Trains the clients independently for 9 local epochs over 5 federated communication rounds.
   * Uses different optimizers across clients: AdamW, SGD, and RMSprop.
   * Aggregates local model updates using Federated Averaging (FedAvg) without transferring raw medical images.
3. Client-Specific Optimization
   Client 0: AdamW, learning rate = 0.0001, weight decay = 1 × 10⁻².
   Client 1:SGD, learning rate = 0.01, momentum = 0.9.
   Client 2:RMSprop, learning rate = 0.0005, α = 0.99.
   Each client locally trains the shared ResNet-34 model and sends the learned model parameters/updates to the central server.
4. Global Model Aggregation & Evaluation
   Combines client model updates using FedAvg after each communication round.
   Evaluates the resulting global model on training, validation, and testing datasets.
   Measures accuracy, misclassification rate, precision, sensitivity, specificity, FPR, FNR, NPV, and F1-score.
   Generates confusion matrices, ROC curves, and Precision-Recall curves for the four leukemia classes.
5. Explainable AI
   Applies Grad-CAM, Grad-CAM++, and Advanced Leukemia Focus to visualize important image regions.
   Uses LIME to identify influential image regions through superpixel-based explanations and leukemia-cell highlighting.
   Uses SHAP to generate feature-level attribution maps and highlight cellular regions contributing to model predictions.
   Provides complementary visual and feature-level explanations of the global model's decisions.
6. RealTime Leukemia Prediction
   Processes a newly acquired microscopic blood-smear image using the same preprocessing pipeline.
   Passes the processed image to the validated global ResNet-34 model.
   Predicts one of the four leukemia-related classes.
   Generates corresponding XAI explanations using Grad-CAM, LIME, and SHAP.
   Stores the diagnostic output and processing information in the cloud for future retrieval, monitoring, and analysis.
All model development and federated experiments are implemented in **Python and PyTorch**, with evaluation and visualization performed using appropriate scientific and machine-learning libraries.

## Usage Instructions
1. Dataset Preparation
   Download the publicly available Multi Cancer Dataset from Kaggle.
   Organize the relevant images into the following directory structure:
   /path/to/dataset/
      ├── All_Benign/
      ├── All_Early/
      ├── All_Pre/
      └── All_Pro/
2. Configure Dataset Path
   Update the dataset path in the Python script or Colab notebook:
3. Run Federated Training
   Execute the federated learning implementation using Python or open the corresponding Google Colab notebook.
   The training process initializes the ResNet-34 model, distributes it to the three clients, performs local training, and aggregates the client updates using FedAvg.
4. Evaluate the Global Model
   Evaluate the aggregated global model on the training, validation, and testing subsets.
   The evaluation generates classification metrics, confusion matrices, ROC curves, Precision-Recall curves, and federated convergence results.
5. Generate XAI Visualizations
   Apply Grad-CAM/Grad-CAM++, LIME, and SHAP to representative test images.
   Generate heatmaps and attribution maps to visualize the image regions contributing to the predicted leukemia class.
6. Real Time Inference
   Provide a new microscopic blood-smear image as input.
   Apply the trained preprocessing pipeline.
   Load the validated global model and obtain the predicted class.
   Generate the corresponding XAI explanations for the prediction.
Usage Instructions
1. Dataset Preparation     
Organize the dataset directory as:
   ├── All_Benign/
   ├── All_Early/
   ├── All_Pre/
   └── All_Pro/
2. Run Training and Evaluation
python Federated_Leukemia_ResNet34.py
Alternatively, open and run the corresponding Google Colab.
The training pipeline performs:
Local model training across three simulated institutional clients.
Client-specific optimization using AdamW, SGD, and RMSprop.
Federated aggregation using FedAvg.
Five communication rounds with nine local epochs per client.
Global model evaluation on the training, validation, and test sets.
3. Outputs Generated
Training, validation, and test accuracy/loss results.
Global model classification report.
Confusion matrices for training, validation, and test datasets.
ROC curves and AUC analysis.
Precision–Recall curves.
Class-wise performance metrics.
Grad-CAM and Grad-CAM++ explainability heatmaps.
Advanced Leukemia Focus visualizations.
LIME-based local explanations.
SHAP-based feature attribution visualizations.
Federated global model (.pth) for subsequent inference and evaluation.
Requirements
Library	Version (Recommended)
|Python	|3.9|
|PyTorch | 2.1|
torchvision|0.16|
NumPy|1.25|
scikit-learn|1.3|
Matplotlib|3.8|
Seaborn	|0.12|
Pillow	|10.0|
SHAP	|0.44|
LIME	|0.2|

Install Dependencies
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow shap lime
|Methodology| Summary|
Component|	Description|
Classification Task|	Four-class acute lymphoblastic leukemia (ALL) classification
Dataset	Balanced leukemia blood-smear image dataset with 20,000 images
Classes		|All_Benign, All_Early, All_Pre, All_Pro|
Data Split	|70% training / 15% validation / 15% testing|
Input Size	|224 × 224 × 3 RGB images|
Base Model	|Pre-trained ResNet-34|
Federated Setup	|Three simulated institutional clients|
Local Training	9 epochs per communication round
Communication Rounds	5
Client 1 Optimizer	AdamW, learning rate = 0.0001, weight decay = 0.01
Client 2 Optimizer	SGD, learning rate = 0.01, momentum = 0.9
Client 3 Optimizer	RMSprop, learning rate = 0.0005, alpha = 0.99
Aggregation		Federated Averaging (FedAvg)
Privacy Strategy	Raw medical images remain local; only model updates are shared
Explainability		Grad-CAM, Grad-CAM++, ALF, LIME, and SHAP
Evaluation Metrics	Accuracy, MCR, Precision, Sensitivity, Specificity, FPR, FNR, NPV, and F1-Score

Performance Summary
Evaluation Phase	|Accuracy| 	|MCR|	|F1-Score|
Training		99.71%		0.29%	0.9971
Testing			99.40%		0.60%	0.9940
Validation		99.67%		0.33%	0.995

Additional test-set performance:
Precision:   99.40%	
Specificity: 99.80%
False Positive Rate: 0.20%
False Negative Rate: 0.60%
ROC-AUC: 1.00 across the evaluated classes

These results are based on the experimental dataset and should not be interpreted as evidence of clinical performance without independent external validation.

Visualization Examples
Confusion Matrices – class-wise prediction performance for training, validation, and testing.
ROC Curves – class-wise discrimination performance and AUC analysis.
Precision–Recall Curves – precision and recall behavior across classification thresholds.
Grad-CAM / Grad-CAM++ – visualization of image regions contributing to model predictions.
Advanced Leukemia Focus (ALF) – focused visualization of leukemia-relevant morphological regions.
LIME – local image explanations using superpixel-based feature attribution.
SHAP – feature attribution highlighting image regions contributing to the predicted class.
Evaluation Environment
Developed and tested on:
- OS: macOS / Windows 10  
- CPU/GPU: Intel Core i7 
- RAM: 16 GB  
- Framework: PyTorch  
Developed and tested using:
Platform: Google Colab / cloud-based simulation environment
Framework: PyTorch
Programming Language: Python
Model Architecture: ResNet-34
Federated Learning: Three-client FedAvg simulation
Visualization: Matplotlib and Seaborn
Explainable AI: Grad-CAM, Grad-CAM++, ALF, LIME, and SHAP
Results and Discussion
The proposed federated ResNet-34 framework achieved 99.40% test accuracy while maintaining strong performance across the four leukemia classes.
Federated averaging enabled collaborative model training across three simulated institutional clients without requiring direct sharing of raw blood-smear images.
The use of different client optimizers—AdamW, SGD, and RMSprop—supports heterogeneous local training conditions within the federated setting.
The XAI component provides complementary explanations through gradient-based, perturbation-based, and feature-attribution approaches.
Grad-CAM, Grad-CAM++, and ALF provide visual localization of discriminative morphological regions, while LIME and SHAP provide additional local attribution perspectives.
The framework combines distributed learning with explainability to support transparency and facilitate analysis of model decisions.
External validation on independent datasets and clinical settings remains necessary to establish generalizability and clinical applicability.
Limitations
The study uses a four-class leukemia classification taxonomy and may not represent the full range of hematological malignancies.
The experimental dataset is balanced and may not fully reflect class distributions encountered in real clinical environments.
The federated learning setup is simulated using three clients rather than deployed across physically independent hospitals or diagnostic laboratories.
The study uses five communication rounds; larger-scale federated deployments may introduce additional communication and computational overhead.
The reported performance requires external validation using independent datasets acquired from different institutions, imaging systems, and staining conditions.
Conclusion
The proposed Federated Deep Learning Framework for Leukemia Diagnosis Using Multi-Level Explainability integrates:
•ResNet-34-based four-class leukemia classification
•Three-client federated learning using FedAvg
•Client-specific optimization using AdamW, SGD, and RMSprop
•Privacy-preserving distributed training without direct sharing of raw images
•Multi-level XAI using Grad-CAM, Grad-CAM++, ALF, LIME, and SHAP
•Comprehensive classification and explainability evaluation
The framework demonstrates how federated deep learning and multi-level explainability can be combined to develop a more transparent approach to automated leukemia image classification. Further validation using independent clinical datasets, larger federated networks, and real-world institutional deployments is required before clinical use




