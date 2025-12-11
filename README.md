# Breast Cancer Polygenic Risk Score Calculator

## Overview

This project implements a Polygenic Risk Score (PRS) system that aggregates the effects of multiple genetic variants (SNPs) identified through genome-wide association studies (GWAS) to estimate breast cancer risk. The model combines genetic information with clinical factors like age and family history for comprehensive risk assessment.

## Features

- **10 validated breast cancer risk SNPs** from published GWAS studies (FGFR2, TOX3, MAP3K1, etc.)
- **Synthetic data generation** based on Hardy-Weinberg equilibrium
- **Logistic regression model** with cross-validation
- **Comprehensive risk metrics**: percentile ranking, relative risk, absolute lifetime risk
- **Visualizations**: PRS distributions, ROC curves, feature importance plots
- **Performance evaluation**: AUC-ROC, sensitivity, specificity, confusion matrices


```python
from main import PolygeneticRiskScore

# Initialize calculator
prs_calc = PolygeneticRiskScore()

# Generate synthetic data
data = prs_calc.generate_genotype_data(n_samples=2000)

# Train model
metrics = prs_calc.train_model(data)

# Calculate individual risk
risk_metrics = prs_calc.calculate_risk_metrics(
    prs=1.8,
    age=55,
    family_history=True
)

print(f"Risk Category: {risk_metrics['risk_category']}")
print(f"Lifetime Risk: {risk_metrics['absolute_risk']:.1f}%")
```

## Output Files

- `genotype_data.csv` - Synthetic patient genotype data
- `snp_database.csv` - SNP reference information
- `prs_distribution.png` - Risk score distribution
- `feature_importance.png` - Variant contribution analysis
- `roc_curve.png` - Model performance visualization

## Risk Interpretation

- **Low Risk** (<20th percentile): Below average genetic risk
- **Average Risk** (20-60th percentile): Population average
- **Moderate-High Risk** (60-80th percentile): Elevated risk, consider enhanced screening
- **High Risk** (>80th percentile): Significantly elevated risk, discuss with physician

## Important Notes

 **This is an educational/research tool** - not for clinical diagnosis

- PRS should be combined with clinical assessment and family history
- Does not replace testing for high-risk mutations (BRCA1/2)
- Effect sizes are based on European populations and may differ across ancestries
- Real clinical applications require validated testing and genetic counseling

## References

- Michailidou et al. (2017). Association analysis identifies 65 new breast cancer risk loci. Nature.
- Mavaddat et al. (2019). Polygenic Risk Scores for Breast Cancer. JAMA Oncology.
- Zhang et al. (2020). Genome-wide association study identifies 32 novel breast cancer susceptibility loci. Nature Genetics.
