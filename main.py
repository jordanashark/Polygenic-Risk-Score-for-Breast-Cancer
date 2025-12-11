"""
Breast Cancer Polygenic Risk Score (PRS) Calculator
Uses known GWAS variants to calculate personalized breast cancer risk
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Known breast cancer risk SNPs from GWAS studies
BREAST_CANCER_SNPS = [
    {'id': 'rs2981582', 'gene': 'FGFR2', 'risk_allele': 'T', 'effect': 0.26, 'maf': 0.38},
    {'id': 'rs3803662', 'gene': 'TOX3', 'risk_allele': 'T', 'effect': 0.20, 'maf': 0.25},
    {'id': 'rs889312', 'gene': 'MAP3K1', 'risk_allele': 'C', 'effect': 0.13, 'maf': 0.28},
    {'id': 'rs13281615', 'gene': '8q24', 'risk_allele': 'G', 'effect': 0.11, 'maf': 0.40},
    {'id': 'rs3817198', 'gene': 'LSP1', 'risk_allele': 'C', 'effect': 0.08, 'maf': 0.30},
    {'id': 'rs13387042', 'gene': '2q35', 'risk_allele': 'A', 'effect': 0.12, 'maf': 0.51},
    {'id': 'rs4973768', 'gene': 'SLC4A7', 'risk_allele': 'T', 'effect': 0.09, 'maf': 0.46},
    {'id': 'rs6504950', 'gene': 'STXBP4', 'risk_allele': 'G', 'effect': 0.07, 'maf': 0.23},
    {'id': 'rs10941679', 'gene': '5p12', 'risk_allele': 'G', 'effect': 0.19, 'maf': 0.27},
    {'id': 'rs2107425', 'gene': 'H19', 'risk_allele': 'T', 'effect': 0.10, 'maf': 0.33}
]


class PolygeneticRiskScore:
    """Calculate and analyze polygenic risk scores for breast cancer"""
    
    def __init__(self, snps: List[Dict] = None):
        """Initialize with SNP data"""
        self.snps = snps if snps else BREAST_CANCER_SNPS
        self.model = None
        
    def generate_genotype_data(self, n_samples: int, case_control_ratio: float = 0.5) -> pd.DataFrame:
        """
        Generate synthetic genotype data based on Hardy-Weinberg equilibrium
        
        Args:
            n_samples: Number of samples to generate
            case_control_ratio: Proportion of cases (breast cancer patients)
            
        Returns:
            DataFrame with genotype data and case/control labels
        """
        data = {'sample_id': [f'SAMPLE_{i:04d}' for i in range(n_samples)]}
        
        # Generate genotypes for each SNP
        for snp in self.snps:
            genotypes = []
            maf = snp['maf']
            
            # Hardy-Weinberg equilibrium probabilities
            p_aa = (1 - maf) ** 2  # homozygous reference
            p_ab = 2 * maf * (1 - maf)  # heterozygous
            p_bb = maf ** 2  # homozygous alternate
            
            for _ in range(n_samples):
                rand = np.random.random()
                if rand < p_aa:
                    genotype = 0  # no risk alleles
                elif rand < p_aa + p_ab:
                    genotype = 1  # one risk allele
                else:
                    genotype = 2  # two risk alleles
                genotypes.append(genotype)
            
            data[snp['id']] = genotypes
        
        df = pd.DataFrame(data)
        
        # Calculate PRS for each sample
        df['prs'] = self.calculate_prs(df)
        
        # Generate case/control status based on PRS
        # Higher PRS = higher probability of being a case
        prs_normalized = (df['prs'] - df['prs'].mean()) / df['prs'].std()
        case_prob = 1 / (1 + np.exp(-prs_normalized))
        
        # Adjust to achieve desired case/control ratio
        threshold = np.percentile(case_prob, (1 - case_control_ratio) * 100)
        df['status'] = (case_prob > threshold).astype(int)
        
        # Add clinical factors
        df['age'] = np.random.randint(30, 80, n_samples)
        df['family_history'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        
        return df
    
    def calculate_prs(self, genotype_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate polygenic risk score for each sample
        
        Args:
            genotype_data: DataFrame with genotype columns
            
        Returns:
            Array of PRS values
        """
        prs = np.zeros(len(genotype_data))
        
        for snp in self.snps:
            if snp['id'] in genotype_data.columns:
                prs += genotype_data[snp['id']].values * snp['effect']
        
        return prs
    
    def calculate_risk_metrics(self, prs: float, age: int, family_history: bool) -> Dict:
        """
        Calculate comprehensive risk metrics for an individual
        
        Args:
            prs: Polygenic risk score
            age: Age in years
            family_history: Whether there's first-degree family history
            
        Returns:
            Dictionary with risk metrics
        """
        # Add clinical factors
        age_factor = 0.3 if age > 50 else 0
        family_factor = 0.5 if family_history else 0
        total_score = prs + age_factor + family_factor
        
        # Calculate percentile (assuming mean PRS ~1.5, SD ~0.5)
        percentile = 50 + (total_score - 1.5) * 20
        percentile = np.clip(percentile, 1, 99)
        
        # Risk category
        if percentile >= 80:
            risk_category = 'High'
        elif percentile >= 60:
            risk_category = 'Moderate-High'
        elif percentile <= 20:
            risk_category = 'Low'
        else:
            risk_category = 'Average'
        
        # Lifetime risk (average baseline is ~12.5%)
        baseline_risk = 12.5
        relative_risk = np.exp(total_score - 1.5)
        absolute_risk = min(50, baseline_risk * relative_risk)
        
        return {
            'prs': prs,
            'total_score': total_score,
            'percentile': percentile,
            'risk_category': risk_category,
            'relative_risk': relative_risk,
            'absolute_risk': absolute_risk,
            'clinical_factors': {
                'age_contribution': age_factor,
                'family_history_contribution': family_factor
            }
        }
    
    def train_model(self, genotype_data: pd.DataFrame) -> Dict:
        """
        Train logistic regression model for breast cancer prediction
        
        Args:
            genotype_data: DataFrame with genotype and clinical data
            
        Returns:
            Dictionary with model performance metrics
        """
        # Prepare features
        snp_cols = [snp['id'] for snp in self.snps]
        X = genotype_data[snp_cols + ['age', 'family_history']].values
        y = genotype_data['status'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'auc_roc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'accuracy': report['accuracy'],
            'sensitivity': report['1']['recall'],
            'specificity': report['0']['recall'],
            'precision': report['1']['precision'],
            'confusion_matrix': cm,
            'feature_importance': dict(zip(
                snp_cols + ['age', 'family_history'],
                np.abs(self.model.coef_[0])
            ))
        }
        
        return metrics
    
    def plot_risk_distribution(self, genotype_data: pd.DataFrame, save_path: str = None):
        """Plot PRS distribution for cases vs controls"""
        plt.figure(figsize=(10, 6))
        
        cases = genotype_data[genotype_data['status'] == 1]['prs']
        controls = genotype_data[genotype_data['status'] == 0]['prs']
        
        plt.hist(controls, bins=30, alpha=0.6, label='Controls', color='blue')
        plt.hist(cases, bins=30, alpha=0.6, label='Cases', color='red')
        
        plt.xlabel('Polygenic Risk Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Polygenic Risk Scores', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, metrics: Dict, save_path: str = None):
        """Plot feature importance from trained model"""
        importance = metrics['feature_importance']
        features = list(importance.keys())
        values = list(importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(values)[::-1]
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), values, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.title('Feature Importance in PRS Model', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, genotype_data: pd.DataFrame, save_path: str = None):
        """Plot ROC curve for the trained model"""
        from sklearn.metrics import roc_curve
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        snp_cols = [snp['id'] for snp in self.snps]
        X = genotype_data[snp_cols + ['age', 'family_history']].values
        y = genotype_data['status'].values
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Breast Cancer PRS Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function demonstrating the PRS calculator"""
    
    print("=" * 70)
    print("Breast Cancer Polygenic Risk Score Calculator")
    print("=" * 70)
    print()
    
    # Initialize PRS calculator
    prs_calc = PolygeneticRiskScore()
    
    # Generate synthetic data
    print("Generating synthetic genotype data...")
    n_samples = 2000
    data = prs_calc.generate_genotype_data(n_samples, case_control_ratio=0.3)
    print(f"Generated {n_samples} samples")
    print(f"Cases: {data['status'].sum()}, Controls: {(1-data['status']).sum()}")
    print()
    
    # Train model
    print("Training logistic regression model...")
    metrics = prs_calc.train_model(data)
    print()
    print("Model Performance Metrics:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print()
    
    # Example patient risk calculation
    print("=" * 70)
    print("Example Patient Risk Assessment")
    print("=" * 70)
    
    # Simulate a patient
    patient_genotypes = {snp['id']: np.random.choice([0, 1, 2]) for snp in BREAST_CANCER_SNPS}
    patient_df = pd.DataFrame([patient_genotypes])
    patient_prs = prs_calc.calculate_prs(patient_df)[0]
    
    patient_age = 55
    patient_family_history = True
    
    risk_metrics = prs_calc.calculate_risk_metrics(patient_prs, patient_age, patient_family_history)
    
    print(f"\nPatient Information:")
    print(f"  Age: {patient_age} years")
    print(f"  Family History: {'Yes' if patient_family_history else 'No'}")
    print(f"\nGenetic Risk Scores:")
    print(f"  Polygenic Risk Score: {risk_metrics['prs']:.3f}")
    print(f"  Total Risk Score: {risk_metrics['total_score']:.3f}")
    print(f"\nRisk Assessment:")
    print(f"  Risk Category: {risk_metrics['risk_category']}")
    print(f"  Percentile: {risk_metrics['percentile']:.1f}th")
    print(f"  Relative Risk: {risk_metrics['relative_risk']:.2f}x")
    print(f"  Absolute Lifetime Risk: {risk_metrics['absolute_risk']:.1f}%")
    print(f"  (Population average: ~12.5%)")
    print()
    
    # Show top contributing SNPs
    print("Top Contributing SNPs:")
    contributions = []
    for snp in BREAST_CANCER_SNPS:
        genotype = patient_genotypes[snp['id']]
        contrib = genotype * snp['effect']
        contributions.append((snp['id'], snp['gene'], genotype, contrib))
    
    contributions.sort(key=lambda x: x[3], reverse=True)
    for snp_id, gene, genotype, contrib in contributions[:5]:
        print(f"  {snp_id} ({gene}): genotype={genotype}, contribution={contrib:.3f}")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    prs_calc.plot_risk_distribution(data, 'prs_distribution.png')
    prs_calc.plot_feature_importance(metrics, 'feature_importance.png')
    prs_calc.plot_roc_curve(data, 'roc_curve.png')
    print("Plots saved!")
    print()
    
    # Save results
    print("Saving results...")
    data.to_csv('genotype_data.csv', index=False)
    
    # Save SNP information
    snp_df = pd.DataFrame(BREAST_CANCER_SNPS)
    snp_df.to_csv('snp_database.csv', index=False)
    
    print("Analysis complete!")
    print("Files saved:")
    print("  - genotype_data.csv: Synthetic genotype data")
    print("  - snp_database.csv: SNP information")
    print("  - prs_distribution.png: PRS distribution plot")
    print("  - feature_importance.png: Feature importance plot")
    print("  - roc_curve.png: ROC curve")


if __name__ == "__main__":
    main()