\# Stage E Step 3 — MLflow Training Plan



\## Goal



Add MLflow tracking for model training and candidate model creation.



The system should support:



```text

train → evaluate → log to MLflow → save candidate → compare with champion → promote only if better

