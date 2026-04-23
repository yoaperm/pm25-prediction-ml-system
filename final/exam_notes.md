# สรุปวิชา Software Engineering for ML Systems
## โน้ตอ่านสอบ (Exam Notes) - Bullet Point

---

## 1. Introduction to ML Systems (สัปดาห์ 1)
### ML Systems vs Traditional Software
- **ML Systems** = Model + Data + Config + Code + Infrastructure
- ความท้าทาย: ≤30% code is ML, ≥70% technical debt จาก data prep, feature engineering, infrastructure
- **Technical Debt in ML**: 
  - Data dependency (ต้องฟอร์แมทเดิม ตลอดเวลา)
  - Model dependency (เปลี่ยนแบบหนึ่ง อาจกระทบการทำงาน)
  - Boundary erosion (ขอบเขตของ ML และ non-ML เบลอ)

### ML Pipeline Workflow
1. Problem Definition → 2. Data Collection → 3. Feature Engineering → 4. Model Training → 5. Evaluation → 6. Deployment → 7. Monitoring → 8. Retrain

---

## 2. Building ML Systems From Models to Products (สัปดาห์ 2)
### System Design Framework
- **Scope**: Business requirements, user needs, constraints (latency, cost, accuracy)
- **Key Questions**:
  - Model type? (Classification, Regression, NLP, Computer Vision)
  - Online vs Batch prediction?
  - Real-time or offline learning?
  - Cold start? (ยังไม่มีข้อมูลต้องทำยังไง)

### ML System Architecture
```
Data Ingestion → Data Storage → Feature Store → Model Training → Model Registry → Inference Service → Monitoring
```
- **Feature Store**: Central hub for reusable, versioned features
- **Model Registry**: Track models, versions, metadata, staging/production status
- **Inference Service**: API/Batch serving, model loading, latency optimization

### Model Serving Patterns
- **Batch**: High throughput, scheduling-based (e.g., nightly)
- **Real-time**: Low latency, synchronous API calls
- **Streaming**: Continuous data, windowed aggregations
- **Hybrid**: Mix of batch + real-time

---

## 3. Data Engineering Fundamentals (สัปดาห์ 3)
### Data Pipeline Architecture
- **Collection**: Logs, APIs, sensors, databases
- **Storage**: Raw data lake (immutable), processed warehouse (clean)
- **Quality**: Validation, schema enforcement, anomaly detection
- **Lineage**: Track data origin → transformation → usage

### Key Data Issues
- **Missing Data**: MCAR (Missing Completely At Random), MAR (Missing At Random), MNAR (Missing Not At Random)
  - Solutions: deletion, imputation (mean/median/forward fill), prediction-based
- **Outliers**: Detect (IQR, Z-score, Isolation Forest), handle (remove/cap/transform)
- **Imbalanced Data**: Resampling (oversampling, undersampling), SMOTE, class weights

### Data Versioning (DVC - Data Version Control)
- Track data like code (branches, commits, tags)
- Reproducible datasets, experiment tracking
- Integration with ML pipelines

---

## 4. Mastering Training Data (สัปดาห์ 3)
### Data Labeling & Annotation
- **Manual Labeling**: Slow, expensive, but accurate (baseline)
- **Active Learning**: Select hardest examples to label (minimize labeling cost)
- **Weak Supervision**: Use heuristics, rules, crowdsourcing
- **Data Augmentation**: Rotate, flip, noise, mixup (images); synonym replacement (text)

### Train/Val/Test Split
- **Time Series**: Train [old] → Val [recent] → Test [newest] (no future leakage)
- **Stratified**: Maintain class distribution (for imbalanced data)
- **K-Fold CV**: Multiple splits reduce variance

### Data Quality Metrics
- **Completeness**: % non-null values
- **Consistency**: Similar values in different sources?
- **Accuracy**: Ground truth correctness
- **Timeliness**: Data freshness

---

## 5. Feature Engineering (สัปดาห์ 4)
### Feature Types & Transformations
- **Numerical**: Normalization (StandardScaler), Scaling (MinMaxScaler), Log transform
- **Categorical**: One-hot encoding, Label encoding, Embedding
- **Temporal**: Lag features, Rolling stats, Trend, Seasonality, Time-based (hour, day, month)
- **Domain-specific**: Domain knowledge beats generic features

### Feature Design Best Practices
- **Leakage Prevention**: 
  - Information from future = leakage (train on past only)
  - Target leakage: Feature directly contains info about target
  - Example: Time series forecasting must shift features by at least 1 timestep
- **Feature Scaling**: Essential for distance-based models (KNN, SVM), tree-based models don't care
- **Feature Selection**: Remove correlated (multicollinearity), low-variance, irrelevant
  - Methods: Correlation matrix, permutation importance, SHAP values

### Feature Stores
- Centralized, versioned, reusable features
- Consistency: Same feature computation for training & inference
- Example: Feast, Tecton, Databricks Feature Store

---

## 6. Model Development & Offline Evaluation (สัปดาห์ 4)
### Model Selection
- **Regression**: Linear, Ridge/Lasso, Random Forest, XGBoost, SVR, Neural Networks
- **Classification**: Logistic Regression, SVM, Random Forest, Gradient Boosting, Neural Networks
- **Time Series**: ARIMA, LSTM, Transformer, XGBoost (with lag features)
- **Trade-offs**: Interpretability vs Accuracy, Training time vs Performance

### Hyperparameter Tuning
- **Grid Search**: Try all combinations (exhaustive, slow for many params)
- **Random Search**: Random sampling (faster, may miss optimal)
- **Bayesian Optimization**: Probabilistic (efficient, best for expensive models)
- **Validation Strategy**: K-Fold CV (for i.i.d data), Time Series Split (for temporal data)

### Evaluation Metrics
- **Regression**: MAE, RMSE, R², MAPE
  - MAE = Mean Absolute Error (avg |y_true - y_pred|) — **robust to outliers**
  - RMSE = Root Mean Squared Error (penalizes large errors more) — **sensitive to outliers**
  - R² = coefficient of determination (% variance explained)
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - **Confusion Matrix**: TP, FP, TN, FN
  - Precision = TP/(TP+FP) — "of predicted positive, how many correct?"
  - Recall = TP/(TP+FN) — "of actual positive, how many detected?"
  - F1 = 2 * (Precision * Recall) / (Precision + Recall) — harmonic mean
- **Imbalanced Data**: Use F1, ROC-AUC, PR-AUC (not just accuracy)

### Bias-Variance Trade-off
- **Underfitting** (high bias): Model too simple, poor on train & test
- **Overfitting** (high variance): Model too complex, good on train, poor on test
- **Solution**: Cross-validation, regularization (L1/L2), early stopping, dropout

---

## 7. Model Deployment & Prediction Service (สัปดาห์ 7)
### Deployment Strategies
- **Batch Inference**: Offline, high throughput (e.g., daily retrain & score)
  - Storage: HDFS, S3, databases
  - Trigger: Scheduler (cron), event-based
- **Real-time Inference**: 
  - Synchronous API (REST, gRPC)
  - Latency requirement (e.g., <100ms for recommendations)
  - Load balancing, auto-scaling
- **Edge Deployment**: Mobile, IoT (model size matters, quantization)

### Model Serving Infrastructure
- **API Server**: FastAPI, Flask, Django (Python); Spring (Java); Express (Node.js)
- **Containerization**: Docker, Kubernetes for orchestration
- **Model Serving Frameworks**:
  - **Triton Inference Server**: Multi-framework (TensorFlow, PyTorch, ONNX), GPU support
  - **KServe**: Kubernetes-native, auto-scaling
  - **Seldon Core**: Model as a service, A/B testing
  - **BentoML**: Python-first, one-command deployment

### Model Export & Format
- **ONNX** (Open Neural Network Exchange): Universal format, cross-platform (Python → Java/Go/C++)
- **SavedModel** (TensorFlow): Native format
- **Pickle**: Python-specific (security risk: arbitrary code execution)
- **Benefits**: Model portability, production reproducibility, inference optimization

### Handling Predictions
- **Prediction Pipeline**: Load model → preprocess input → predict → post-process
- **Feature Consistency**: Same preprocessing for train/inference (feature store solves this)
- **Latency Optimization**: 
  - Model quantization (float32 → int8)
  - Caching frequent inputs
  - Batch processing
  - Model distillation (smaller model, slightly less accurate)
- **Serving Cost**: Trade-off between latency, throughput, & infrastructure cost

---

## 8. Data Distribution Shifts & Monitoring (สัปดาห์ 7)
### Types of Data Shift
1. **Covariate Shift**: P(X) changes, but P(Y|X) stays same
   - Example: Summer vs winter traffic patterns
2. **Label Shift**: P(Y) changes, but P(X|Y) stays same
   - Example: Class imbalance ratio shifts
3. **Concept Drift**: P(Y|X) changes (most dangerous)
   - Example: Model from 2020 doesn't work in 2025
4. **Prior Shift**: Combination of covariate + label shift

### Drift Detection Methods
- **Statistical Tests**: KL divergence, Kolmogorov-Smirnov, Jensen-Shannon distance
- **Population Stability Index (PSI)**: Measures distributional shift
  - PSI = Σ (% new - % old) * ln(% new / % old)
  - PSI < 0.1: Minimal shift, PSI > 0.25: Significant shift
- **Model Performance Monitoring**: Track MAE, precision, recall, AUC over time
- **Data Quality Metrics**: Check for missing values, outliers, schema violations

### Monitoring Metrics
- **Model Performance**: Accuracy degradation, latency, throughput
- **Input Data**: Distribution of features, missing rates
- **Predictions**: Distribution shift, prediction confidence
- **System Health**: API uptime, error rates, resource utilization

### Retraining Strategies
- **Scheduled**: Fixed interval (weekly, monthly)
- **Reactive**: Triggered by drift detection (MAE > threshold)
- **Incremental**: Add new data to old training set
- **Full Retrain**: Rebuild from scratch (reset parameters)

---

## 9. ML Infrastructure & Tooling (MLOps) (สัปดาห์ 6)
### Orchestration & Scheduling
- **Apache Airflow**: DAG-based, complex dependencies, Python-native
  - Task: Discrete unit of work
  - DAG (Directed Acyclic Graph): Task order & dependencies
  - Scheduler: Runs tasks on time, retries on failure
- **Workflow Management**: Error handling, retry logic, monitoring
- **Example DAG**: Extract data → Clean → Feature eng. → Train → Deploy → Monitor

### ML Experiment Tracking
- **MLflow**: 
  - **Tracking**: Log params, metrics, artifacts
  - **Models**: Version, stage (staging/production)
  - **Registry**: Central model hub
  - **Projects**: Packaging reproducible runs
- **Benefits**: Reproducibility, comparison, collaboration

### Data Pipeline Tools
- **DVC (Data Version Control)**: Version data, pipelines, experiments
- **Great Expectations**: Data quality validation
- **Apache Spark**: Distributed data processing (big data)

### Infrastructure as Code (IaC)
- **Docker**: Containerize models + dependencies
- **Kubernetes**: Orchestrate containers, auto-scaling, service discovery
- **Terraform/CloudFormation**: Provision cloud resources
- **Benefits**: Reproducibility, scalability, multi-environment consistency

---

## 10. Process & Team (Organization) (สัปดาห์ 8)
### ML Project Roles
- **Data Engineer**: Data collection, pipeline, quality
- **ML Engineer**: Model training, optimization, deployment
- **ML Ops/Platform Engineer**: Infrastructure, monitoring, tooling
- **Domain Expert**: Business requirements, feature validation

### ML Project Lifecycle
1. **Scoping**: Business problem, constraints, success metrics
2. **Data Preparation**: Collection, labeling, validation
3. **Model Development**: Experimentation, iteration
4. **Deployment**: Testing, monitoring setup
5. **Maintenance**: Performance tracking, retraining

### Best Practices
- **Reproducibility**: Version code, data, configs, requirements
- **Testing**: 
  - **Unit Tests**: Individual functions (preprocessing, feature engineering)
  - **Integration Tests**: Pipeline end-to-end
  - **Model Tests**: Accuracy, prediction ranges, edge cases
- **Documentation**: README, API docs, data dictionaries, architecture diagrams
- **Code Review**: Pair programming, pull requests
- **CI/CD**: Automated testing, deployment pipelines

---

## 11. Responsible ML Engineering (สัปดาห์ 9)
### Fairness & Bias
- **Sources of Bias**: 
  - Training data (underrepresented groups)
  - Labeling process (annotator bias)
  - Model architecture (inherent to algorithm)
  - Evaluation (metrics not capturing fairness)
- **Fairness Definitions**:
  - **Demographic Parity**: Predictions equal across groups
  - **Equalized Odds**: Equal TPR & FPR across groups
  - **Calibration**: Prediction probability = actual rate
- **Mitigation**: 
  - Balanced training data
  - Fairness metrics in evaluation
  - Threshold tuning per group
  - Post-processing adjustments

### Interpretability & Explainability
- **Why it matters**: Regulatory (GDPR, Fair Lending), user trust, debugging
- **Model-Agnostic Methods**:
  - **SHAP (SHapley Additive exPlanations)**: Feature contribution to prediction
  - **LIME (Local Interpretable Model-agnostic Explanations)**: Local approximation
  - **Permutation Importance**: Feature importance via shuffling
  - **Partial Dependence Plot**: Feature effect on target
- **Intrinsically Interpretable Models**: Linear regression, decision trees (vs black box like deep NN)

### Privacy & Security
- **Data Privacy**:
  - **Differential Privacy**: Add noise to data/gradients (DP-SGD in training)
  - **Federated Learning**: Train on device, share only model updates
  - **Data Anonymization**: Remove PII (personally identifiable info)
- **Model Security**:
  - **Adversarial Attacks**: Tiny perturbations to input fool model
  - **Model Stealing**: Extract model via API queries
  - **Poisoning**: Inject malicious data into training
- **Mitigation**: Input validation, rate limiting, model robustness testing, audit trails

### Model Governance
- **Model Cards**: Document purpose, performance, fairness, limitations
- **Data Sheets**: Document data collection, preprocessing, bias, licensing
- **Ethical Review**: Before deployment (especially high-stakes: hiring, loans, healthcare)

---

## 12. Continual Learning & Testing in Production (สัปดาห์ 9)
### Concept Drift & Model Degradation
- **Online Learning**: Update model continuously (stream data)
- **Batch Retraining**: Periodic retraining on new data
- **Incremental Learning**: Add new data without forgetting old patterns (catastrophic forgetting in NN)

### A/B Testing in ML
- **Setup**: 
  - Control: Old model
  - Treatment: New model
  - Randomize users → measure business metric
- **Metrics**: CTR, conversion rate, user retention, revenue
- **Statistical Significance**: Power analysis, p-value < 0.05
- **Pitfalls**: 
  - Multiple comparisons (alpha inflation)
  - Sample size too small (low power)
  - Interaction effects (new model works differently for subgroups)

### Canary Deployment
- **Gradual Rollout**: 1% traffic → 10% → 100% (monitor for issues)
- **Fallback Plan**: Revert to old model if metrics degrade
- **Shadow Mode**: New model runs in parallel, predictions not used

### Testing Strategy
- **Unit Tests**: Data loaders, preprocessors, loss functions
- **Integration Tests**: E2E pipeline (data → train → predict)
- **Model Tests**: 
  - **Performance Regression**: New model not worse than baseline
  - **Prediction Tests**: Output bounds (e.g., 0-100 for percentage)
  - **Edge Cases**: Rare values, extreme inputs
- **Data Tests**: Schema, missing rates, outliers, distribution
- **Infrastructure Tests**: API latency, throughput, availability

---

## 13. Automation in ML Lifecycle (AutoML & MLOps) (สัปดาห์ 6)
### Automated Machine Learning (AutoML)
- **Goals**: Reduce manual tuning, democratize ML, speed up experimentation
- **Components**:
  - **Hyperparameter Optimization**: Bayesian, bandit algorithms
  - **Neural Architecture Search (NAS)**: Automatic model design
  - **Meta-Learning**: Learn to learn (leverage past experiments)
- **Tools**: AutoKeras, Auto-sklearn, TPOT, H2O AutoML

### MLOps = DevOps for ML
- **Continuous Integration**: Auto-test code, data, models on commit
- **Continuous Training**: Auto-retrain on new data/drift
- **Continuous Deployment**: Auto-push model to production
- **Continuous Monitoring**: Track performance, alert on degradation
- **Key Difference from DevOps**: 
  - Code testing is deterministic (same input → same output)
  - ML testing is probabilistic (randomness in data, training)

### ML Pipeline Automation
- **Infrastructure as Code**: Reproducible environments
- **Feature Pipelines**: Automatic feature engineering, data lineage
- **Model Training Pipelines**: Auto-hyperparameter search, evaluation, comparison
- **Model Serving Pipelines**: Auto-deployment, canary testing, rollback

---

## 14. Advanced Topics
### Transfer Learning
- **Concept**: Pretrained model (e.g., ImageNet) → fine-tune on new task
- **Benefits**: Fast training, small training set, better generalization
- **Types**:
  - **Feature Extraction**: Freeze early layers, train last layers
  - **Fine-tuning**: Train all layers, low learning rate
- **Common Pretrained Models**: BERT (NLP), ResNet (vision), GPT (generation)

### Ensemble Methods
- **Bagging**: Train multiple models on random data subsets (Random Forest)
- **Boosting**: Iteratively train models, emphasize misclassified examples (XGBoost, LightGBM)
- **Stacking**: Combine base models with meta-model
- **Benefits**: Reduced variance, better generalization

### Time Series Forecasting
- **Characteristics**: Temporal dependency, trend, seasonality, cyclicity
- **Methods**:
  - **Classical**: ARIMA, Exponential Smoothing
  - **ML-based**: XGBoost + lag features, LSTM
  - **Hybrid**: ARIMA + NN
- **Challenges**: Non-stationary, concept drift, sparse events
- **Evaluation**: 
  - **Train/Val/Test Split**: Chronological (no future info)
  - **Metrics**: MAE, RMSE, MAPE
  - **Backtesting**: Validate on multiple historical periods

### Recommendation Systems
- **Types**:
  - **Collaborative Filtering**: User-user, item-item similarity
  - **Content-based**: Recommend similar items
  - **Hybrid**: Combine both
- **Challenges**: Cold start, sparsity, popularity bias, diversity
- **Metrics**: Recall@K, NDCG, Hit Rate

---

## 15. Key Takeaways & Exam Focus
### Core Concepts (MUST KNOW)
1. **ML System Architecture**: Data → Features → Model → Prediction → Monitoring
2. **Feature Engineering**: Most important step, prevent leakage (shift 1 timestep)
3. **Train/Val/Test Split**: Proper strategy prevents overfitting & data leakage
4. **Model Evaluation**: Choose metrics matching business goal (accuracy ≠ F1 ≠ AUC)
5. **Deployment Strategy**: Batch vs Real-time, model serving framework, containerization
6. **Monitoring & Retraining**: Detect drift, trigger retraining, versioning
7. **MLOps**: Infrastructure, orchestration, testing, deployment automation

### Common Pitfalls (AVOID)
- ❌ Data leakage (using future info)
- ❌ Overfitting without validation
- ❌ Ignoring class imbalance (using accuracy alone)
- ❌ Deploying without monitoring
- ❌ Not versioning data, code, models
- ❌ Feature inconsistency between train/inference

### Questions to Ask in System Design
1. **Problem**: Regression/Classification? Online/Batch? Real-time?
2. **Data**: Volume? Format? Quality? Labeling cost?
3. **Model**: Accuracy target? Latency/throughput requirement? Explainability needed?
4. **Infrastructure**: Cloud/On-prem? GPU? Distributed?
5. **Monitoring**: What metric indicates failure? Retraining frequency?

---

## 16. Real Project Example: PM2.5 Forecasting System (Hands-On Context)
### Architecture
- **Data**: Bangkok monitoring stations (hourly PM2.5 values)
- **Features**: 19 temporal features (lags, rolling means, time-of-day, seasonality)
- **Models**: 5 competing (Linear, Ridge, RF, XGBoost, LSTM)
- **Training**: GridSearchCV + TimeSeriesSplit (prevent leakage)
- **Evaluation**: MAE primary metric
- **Deployment**: ONNX export → Triton serving
- **Monitoring**: 30-day rolling MAE, PSI for drift detection
- **Auto-retrain**: Trigger if MAE > 6.0 or PSI > 0.25

### Key Decisions Applied
- ✅ **TimeSeriesSplit**: Prevent future leakage in time-series forecasting
- ✅ **shift(1) on all features**: Avoid target leakage
- ✅ **ONNX export**: Model portability & production reproducibility
- ✅ **Triton serving**: GPU inference, multiple models, hot-reload
- ✅ **Monitoring pipeline**: Auto-detect performance degradation
- ✅ **Airflow DAGs**: Orchestrate training, deployment, monitoring

---

## 📝 Quick Reference: Formula & Definitions
| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | Σ\|y - ŷ\| / n | Avg absolute error (robust) |
| **RMSE** | √(Σ(y - ŷ)² / n) | Penalizes large errors |
| **R²** | 1 - (SS_res / SS_tot) | % variance explained |
| **Precision** | TP / (TP + FP) | Correctness of predictions |
| **Recall** | TP / (TP + FN) | Completeness of predictions |
| **F1** | 2 * (P * R) / (P + R) | Harmonic mean (imbalanced) |
| **ROC-AUC** | Area under ROC | Discrimination ability |
| **PSI** | Σ (% new - % old) * ln(%new/%old) | Drift magnitude (>0.25 = significant) |

---

**Last Updated**: April 2026  
**Status**: Ready for Exam  
**Topics Covered**: 16 chapters, 100+ key concepts
