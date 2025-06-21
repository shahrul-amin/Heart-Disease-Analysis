"""
Data Quality Assessment Module
Comprehensive data quality evaluation and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataQualityAssessor:
    """
    Comprehensive data quality assessment tool
    Evaluates completeness, consistency, accuracy, and validity
    """
    
    def __init__(self):
        self.quality_report = {}
        self.recommendations = []
    
    def assess_completeness(self, df):
        """Assess data completeness - missing values analysis"""
        completeness = {}
        
        # Overall completeness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_rate = ((total_cells - missing_cells) / total_cells) * 100
        
        completeness['overall'] = {
            'completeness_rate': completeness_rate,
            'total_cells': total_cells,
            'missing_cells': missing_cells
        }
        
        # Column-wise completeness
        column_completeness = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_rate = (missing_count / len(df)) * 100
            column_completeness[col] = {
                'missing_count': missing_count,
                'missing_rate': missing_rate,
                'completeness_rate': 100 - missing_rate
            }
        
        completeness['columns'] = column_completeness
        
        # Missing value patterns
        missing_patterns = df.isnull().groupby(df.isnull().columns.tolist()).size().sort_values(ascending=False)
        completeness['patterns'] = missing_patterns.head(10).to_dict()
        
        return completeness
    
    def assess_consistency(self, df):
        """Assess data consistency - data type and format consistency"""
        consistency = {}
        
        # Data type consistency
        type_issues = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                sample_types = set()
                for val in df[col].dropna().head(100):
                    sample_types.add(type(val).__name__)
                
                if len(sample_types) > 1:
                    type_issues[col] = list(sample_types)
        
        consistency['data_types'] = type_issues
        
        # Value format consistency (for categorical columns)
        format_issues = {}
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique()
            # Check for case inconsistencies
            if len(unique_vals) > 1:
                lower_vals = [str(val).lower() for val in unique_vals]
                if len(set(lower_vals)) < len(unique_vals):
                    format_issues[col] = "Case inconsistencies detected"
        
        consistency['formats'] = format_issues
        
        # Duplicate records
        duplicates = df.duplicated().sum()
        consistency['duplicates'] = {
            'count': duplicates,
            'percentage': (duplicates / len(df)) * 100
        }
        
        return consistency
    
    def assess_validity(self, df):
        """Assess data validity - domain-specific validation"""
        validity = {}
        
        # Define domain rules for heart disease dataset
        domain_rules = {
            'age': {'min': 0, 'max': 120, 'type': 'numeric'},
            'trestbps': {'min': 50, 'max': 300, 'type': 'numeric'},
            'chol': {'min': 100, 'max': 600, 'type': 'numeric'},
            'thalch': {'min': 60, 'max': 220, 'type': 'numeric'},
            'oldpeak': {'min': 0, 'max': 10, 'type': 'numeric'},
            'ca': {'min': 0, 'max': 4, 'type': 'integer'},
            'num': {'min': 0, 'max': 4, 'type': 'integer'},
            'sex': {'valid_values': ['Male', 'Female', 'M', 'F'], 'type': 'categorical'},
            'fbs': {'valid_values': [0, 1, 'TRUE', 'FALSE'], 'type': 'boolean'},
            'exang': {'valid_values': [0, 1, 'TRUE', 'FALSE'], 'type': 'boolean'}
        }
        
        rule_violations = {}
        
        for col, rules in domain_rules.items():
            if col in df.columns:
                violations = []
                
                if rules['type'] == 'numeric' and col in df.select_dtypes(include=[np.number]).columns:
                    # Check range violations
                    if 'min' in rules:
                        min_violations = (df[col] < rules['min']).sum()
                        if min_violations > 0:
                            violations.append(f"{min_violations} values below minimum ({rules['min']})")
                    
                    if 'max' in rules:
                        max_violations = (df[col] > rules['max']).sum()
                        if max_violations > 0:
                            violations.append(f"{max_violations} values above maximum ({rules['max']})")
                
                elif rules['type'] == 'categorical' and 'valid_values' in rules:
                    # Check invalid categorical values
                    valid_set = set(rules['valid_values'])
                    actual_values = set(df[col].dropna().unique())
                    invalid_values = actual_values - valid_set
                    if invalid_values:
                        violations.append(f"Invalid values: {invalid_values}")
                
                if violations:
                    rule_violations[col] = violations
        
        validity['domain_rules'] = rule_violations
        
        # Statistical validity (outliers)
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'num':  # Skip target variable
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'bounds': [lower_bound, upper_bound]
                }
        
        validity['outliers'] = outlier_info
        
        return validity
    
    def assess_accuracy(self, df):
        """Assess data accuracy - cross-field validation and logical consistency"""
        accuracy = {}
        
        # Logical consistency checks
        logical_issues = []
        
        # Age-related checks
        if 'age' in df.columns and 'thalch' in df.columns:
            # Maximum heart rate should generally decrease with age
            # Simple check: thalch > 220 - age is unusual
            predicted_max_hr = 220 - df['age']
            unusual_hr = df[df['thalch'] > predicted_max_hr + 20]
            if len(unusual_hr) > 0:
                logical_issues.append({
                    'issue': 'Unusually high heart rate for age',
                    'count': len(unusual_hr),
                    'percentage': (len(unusual_hr) / len(df)) * 100
                })
        
        # Blood pressure and age consistency
        if 'age' in df.columns and 'trestbps' in df.columns:
            # Very low blood pressure in elderly might be unusual
            elderly_low_bp = df[(df['age'] > 65) & (df['trestbps'] < 90)]
            if len(elderly_low_bp) > 0:
                logical_issues.append({
                    'issue': 'Low blood pressure in elderly patients',
                    'count': len(elderly_low_bp),
                    'percentage': (len(elderly_low_bp) / len(df)) * 100
                })
        
        accuracy['logical_consistency'] = logical_issues
        
        # Cross-field correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            # Identify unexpectedly high correlations (might indicate data leakage)
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8 and correlation_matrix.columns[i] != 'num':
                        high_correlations.append({
                            'variables': [correlation_matrix.columns[i], correlation_matrix.columns[j]],
                            'correlation': corr_val
                        })
            
            accuracy['high_correlations'] = high_correlations
        
        return accuracy
    
    def generate_recommendations(self, quality_report):
        """Generate actionable recommendations based on quality assessment"""
        recommendations = []
        
        # Completeness recommendations
        if 'completeness' in quality_report:
            overall_completeness = quality_report['completeness']['overall']['completeness_rate']
            if overall_completeness < 95:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Completeness',
                    'issue': f'Overall data completeness is {overall_completeness:.1f}%',
                    'recommendation': 'Investigate missing data patterns and implement appropriate imputation strategies'
                })
            
            # Column-specific recommendations
            for col, info in quality_report['completeness']['columns'].items():
                if info['missing_rate'] > 20:
                    recommendations.append({
                        'priority': 'High',
                        'category': 'Completeness',
                        'issue': f'Column {col} has {info["missing_rate"]:.1f}% missing values',
                        'recommendation': f'Consider dropping {col} or use advanced imputation techniques'
                    })
                elif info['missing_rate'] > 5:
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Completeness',
                        'issue': f'Column {col} has {info["missing_rate"]:.1f}% missing values',
                        'recommendation': f'Apply appropriate imputation for {col} based on its distribution'
                    })
        
        # Validity recommendations
        if 'validity' in quality_report:
            # Domain rule violations
            for col, violations in quality_report['validity']['domain_rules'].items():
                for violation in violations:
                    recommendations.append({
                        'priority': 'High',
                        'category': 'Validity',
                        'issue': f'Domain rule violation in {col}: {violation}',
                        'recommendation': f'Investigate and correct invalid values in {col}'
                    })
            
            # Outlier recommendations
            for col, info in quality_report['validity']['outliers'].items():
                if info['percentage'] > 5:
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Validity',
                        'issue': f'Column {col} has {info["percentage"]:.1f}% outliers',
                        'recommendation': f'Investigate outliers in {col} - consider capping, transformation, or removal'
                    })
        
        # Consistency recommendations
        if 'consistency' in quality_report:
            if quality_report['consistency']['duplicates']['count'] > 0:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Consistency',
                    'issue': f'{quality_report["consistency"]["duplicates"]["count"]} duplicate records found',
                    'recommendation': 'Remove or investigate duplicate records'
                })
            
            for col, issue in quality_report['consistency']['formats'].items():
                recommendations.append({
                    'priority': 'Low',
                    'category': 'Consistency',
                    'issue': f'Format inconsistency in {col}: {issue}',
                    'recommendation': f'Standardize format for {col} values'
                })
        
        # Accuracy recommendations
        if 'accuracy' in quality_report:
            for issue in quality_report['accuracy']['logical_consistency']:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Accuracy',
                    'issue': issue['issue'],
                    'recommendation': 'Review and validate these records with domain experts'
                })
        
        return sorted(recommendations, key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1}[x['priority']], reverse=True)
    
    def run_full_assessment(self, df):
        """Run complete data quality assessment"""
        quality_report = {
            'completeness': self.assess_completeness(df),
            'consistency': self.assess_consistency(df),
            'validity': self.assess_validity(df),
            'accuracy': self.assess_accuracy(df)
        }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(quality_report)
        
        # Calculate overall quality score
        overall_score = self.calculate_quality_score(quality_report)
        
        quality_report['overall_score'] = overall_score
        quality_report['recommendations'] = recommendations
        
        self.quality_report = quality_report
        return quality_report
    
    def calculate_quality_score(self, quality_report):
        """Calculate overall data quality score (0-100)"""
        scores = {}
        
        # Completeness score (40% weight)
        completeness_rate = quality_report['completeness']['overall']['completeness_rate']
        scores['completeness'] = completeness_rate * 0.4
        
        # Validity score (30% weight)
        validity_score = 100  # Start with perfect score
        total_violations = 0
        total_records = 0
        
        for col, violations in quality_report['validity']['domain_rules'].items():
            if violations:
                total_violations += len(violations)
        
        # Reduce score based on violations
        if total_violations > 0:
            validity_score = max(0, 100 - (total_violations * 10))
        
        scores['validity'] = validity_score * 0.3
        
        # Consistency score (20% weight)
        consistency_score = 100
        if quality_report['consistency']['duplicates']['percentage'] > 0:
            consistency_score -= quality_report['consistency']['duplicates']['percentage'] * 2
        
        consistency_score = max(0, consistency_score)
        scores['consistency'] = consistency_score * 0.2
        
        # Accuracy score (10% weight)
        accuracy_score = 100
        logical_issues = len(quality_report['accuracy']['logical_consistency'])
        if logical_issues > 0:
            accuracy_score = max(0, 100 - (logical_issues * 5))
        
        scores['accuracy'] = accuracy_score * 0.1
        
        overall_score = sum(scores.values())
        
        return {
            'overall': overall_score,
            'component_scores': scores,
            'grade': self.get_quality_grade(overall_score)
        }
    
    def get_quality_grade(self, score):
        """Convert quality score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def create_quality_visualizations(self, df):
        """Create visualizations for data quality assessment"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Missing values heatmap
        missing_data = df.isnull()
        sns.heatmap(missing_data, ax=axes[0,0], cbar=True, yticklabels=False, cmap='viridis')
        axes[0,0].set_title('Missing Values Pattern')
        
        # Missing values percentage
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
        if len(missing_pct) > 0:
            missing_pct.plot(kind='barh', ax=axes[0,1])
            axes[0,1].set_title('Missing Values Percentage by Column')
            axes[0,1].set_xlabel('Percentage Missing')
        
        # Data types distribution
        dtype_counts = df.dtypes.value_counts()
        dtype_counts.plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
        axes[1,0].set_title('Data Types Distribution')
        
        # Outliers boxplot for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5 columns
        if len(numeric_cols) > 0:
            df[numeric_cols].boxplot(ax=axes[1,1])
            axes[1,1].set_title('Outliers Detection (First 5 Numeric Columns)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig