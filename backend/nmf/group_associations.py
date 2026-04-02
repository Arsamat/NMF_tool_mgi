#!/usr/bin/env python
#==============================================================================
# File Name :      Module_Group_Associations.py
# Description :    Given a certain K, NMF will produce a matrix of module usage
#                  scores for K modules vs N samples. Given sample groups, assess
#                  how well model usage scores correlate to sample groupings.  
#                  
# Usage :          python3 Module_Group_Associations.py <args>
# Author :         Aura Ferreiro, alferreiro@wustl.edu
# Version :        1.0
# Created On :     2025-08-26
# Last Modified:   2025-08-26
#===============================================================================


"""
Module-Group Association Analysis

This script analyzes the correlation between module usage scores and sample groups
using multiple statistical approaches including ANOVA, effect sizes, and PERMANOVA.

Author: Generated for module usage analysis
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import argparse
import sys
from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

# Optional PERMANOVA import
try:
    from skbio.stats.distance import permanova
    from skbio import DistanceMatrix
    from scipy.spatial.distance import pdist, squareform
    PERMANOVA_AVAILABLE = True
except ImportError:
    PERMANOVA_AVAILABLE = False
    print("Warning: scikit-bio not available. PERMANOVA analysis will be skipped.")


class ModuleGroupAnalyzer:
    """
    A class to analyze associations between module usage scores and sample groups.
    """
    
    def __init__(self, usage_matrix, sample_metadata):
        """
        Initialize the analyzer with usage matrix and sample metadata.
        
        Parameters:
        -----------
        usage_matrix : pd.DataFrame
            Matrix with modules as rows and samples as columns
        sample_metadata : pd.DataFrame
            DataFrame with columns: SampleName, Group
        """
        self.usage_matrix = usage_matrix
        self.sample_metadata = sample_metadata
        self.results = {}
        
        # Validate inputs
        self._validate_inputs()
        
        # Align data
        self._align_data()
    
    def _validate_inputs(self):
        """Validate input data formats and consistency."""
        if not isinstance(self.usage_matrix, pd.DataFrame):
            raise ValueError("usage_matrix must be a pandas DataFrame")
        
        if not isinstance(self.sample_metadata, pd.DataFrame):
            raise ValueError("sample_metadata must be a pandas DataFrame")
        
        required_cols = ['SampleName', 'Group']
        missing_cols = [col for col in required_cols if col not in self.sample_metadata.columns]
        if missing_cols:
            raise ValueError(f"sample_metadata missing required columns: {missing_cols}")
    
    def _align_data(self):
        """Align usage matrix and metadata to have consistent samples."""
        # Get common samples
        matrix_samples = set(self.usage_matrix.index)
        
        meta_samples = set(self.sample_metadata['SampleName'])
        
        common_samples = matrix_samples.intersection(meta_samples)
        
        if len(common_samples) == 0:
            raise ValueError("No common samples between usage matrix and metadata")
        
        print(f"Found {len(common_samples)} common samples out of {len(matrix_samples)} matrix samples "
              f"and {len(meta_samples)} metadata samples")
        
        # Filter and align
        #self.usage_matrix = self.usage_matrix[list(common_samples)]
        self.usage_matrix = self.usage_matrix.loc[list(common_samples)]
        self.sample_metadata = self.sample_metadata[
            self.sample_metadata['SampleName'].isin(common_samples)
        ].set_index('SampleName')
        
        # Ensure same order
        sample_order = self.usage_matrix.index.tolist()
        self.sample_metadata = self.sample_metadata.reindex(sample_order)
    
    def calculate_silhouette_score(self):
        try:
            group_labels = pd.Categorical(self.sample_metadata["Group"]).codes
            silhouette = silhouette_score(self.usage_matrix, group_labels) if len(set(group_labels)) > 1 else 0
        except:
            silhouette = 0
        
        return silhouette
    
    def calculate_module_anova(self, alpha=0.05):
        """
        Perform one-way ANOVA for each module to test group differences.
        
        Parameters:
        -----------
        alpha : float
            Significance level for multiple testing correction
            
        Returns:
        --------
        pd.DataFrame : Results of ANOVA analysis for each module
        """
        print("Performing per-module ANOVA analysis...")
        
        results = []
        groups = self.sample_metadata['Group'].values
        unique_groups = np.unique(groups)
        
        for module in self.usage_matrix.index:
            scores = self.usage_matrix.loc[module].values
            
            # Group data by experimental groups
            group_data = [scores[groups == group] for group in unique_groups]
            
            # Remove groups with no data
            group_data = [g for g in group_data if len(g) > 0]
            
            if len(group_data) < 2:
                # Skip modules with insufficient groups
                results.append({
                    'Module': module,
                    'F_statistic': np.nan,
                    'p_value': np.nan,
                    'eta_squared': np.nan,
                    'mean_score': np.nan,
                    'std_score': np.nan
                })
                continue
            
            # Perform ANOVA
            try:
                f_stat, p_val = f_oneway(*group_data)
                
                # Calculate effect size (eta-squared)
                # η² = SS_between / SS_total
                grand_mean = np.mean(scores)
                ss_total = np.sum((scores - grand_mean) ** 2)
                
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                results.append({
                    'Module': module,
                    'F_statistic': f_stat,
                    'p_value': p_val,
                    'eta_squared': eta_squared,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                })
                
            except Exception as e:
                print(f"Warning: ANOVA failed for module {module}: {e}")
                results.append({
                    'Module': module,
                    'F_statistic': np.nan,
                    'p_value': np.nan,
                    'eta_squared': np.nan,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                })
        
        results_df = pd.DataFrame(results)
        
        # Multiple testing correction
        valid_pvals = ~results_df['p_value'].isna()
        if valid_pvals.sum() > 0:
            reject, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(
                results_df.loc[valid_pvals, 'p_value'], 
                alpha=alpha, 
                method='fdr_bh'
            )
            
            results_df.loc[valid_pvals, 'p_value_corrected'] = pvals_corrected
            results_df.loc[valid_pvals, 'significant'] = reject
        else:
            results_df['p_value_corrected'] = np.nan
            results_df['significant'] = False
        
        # Sort by effect size
        results_df = results_df.sort_values('eta_squared', ascending=False)
        
        self.results['anova'] = results_df
        return results_df
    
    def perform_permanova(self, n_permutations=99, distance_metric='euclidean'):
        """
        Perform PERMANOVA to test multivariate differences between groups.
        
        Parameters:
        -----------
        n_permutations : int
            Number of permutations for the test
        distance_metric : str
            Distance metric to use
            
        Returns:
        --------
        dict : PERMANOVA results
        """
        if not PERMANOVA_AVAILABLE:
            print("PERMANOVA skipped: scikit-bio not available")
            return None
        
        print("Performing PERMANOVA analysis...")
        
        # Transpose matrix (samples as rows, modules as columns)
        data_matrix = self.usage_matrix.T
        data_matrix = data_matrix.reset_index(drop=True)
        groups = self.sample_metadata['Group'].values
        
        # Calculate distance matrix
        data_matrix = data_matrix.apply(pd.to_numeric, errors="coerce")

        # Convert to numpy array for pdist
        X = data_matrix.to_numpy(dtype=float)

        #distances = pdist(X, metric=distance_metric)
        #distance_matrix = DistanceMatrix(squareform(distances), 
         #                              ids=data_matrix.index.astype(str))
        
        distances = pairwise_distances(X, metric=distance_metric, n_jobs=8)
        distance_matrix = DistanceMatrix(distances, ids=[str(i) for i in range(X.shape[0])])
        
        # Perform PERMANOVA
        try:
            results = permanova(distance_matrix, groups, permutations=n_permutations)
            
            # Calculate R2
            F = float(results['test statistic'])
            g = int(len(np.unique(groups)))
            n = int(len(groups))
            df_between = g - 1
            df_within = n - g
            
            # R^2 = SS_between / SS_total = [F * (g-1)] / [F*(g-1) + (n-g)]
            den = (F * df_between + df_within)
            r2 = (F * df_between) / den if den > 0 else 0.0

            # Adjusted R^2 (helps when g is large relative to n)
            r2_adj = 1.0 - ((1.0 - r2) * (n - 1) / (n - g - 1)) if (n - g - 1) > 0 else float("nan")
            
            
            permanova_results = {
                'test_statistic': results['test statistic'],
                'p_value': results['p-value'],
                'n_permutations': results['number of permutations'],
                'distance_metric': distance_metric,
                'df_between': df_between,
                'df_within': df_within,
                'n_groups': g,
                'n_samples': n,
                'r2': float(r2),
                'r2_adj': float(r2_adj)
            }
            
            self.results['permanova'] = permanova_results
            return permanova_results
       
         
        except Exception as e:
            print(f"PERMANOVA failed: {e}")
            return None
    
    def calculate_group_statistics(self):
        """Calculate descriptive statistics for each group and module."""
        print("Calculating group statistics...")
        
        group_stats = []
        
        for module in self.usage_matrix.index:
            scores = self.usage_matrix.loc[module]
            
            for group in self.sample_metadata['Group'].unique():
                group_mask = self.sample_metadata['Group'] == group
                group_scores = scores[group_mask]
                
                group_stats.append({
                    'Module': module,
                    'Group': group,
                    'n_samples': len(group_scores),
                    'mean': np.mean(group_scores),
                    'std': np.std(group_scores),
                    'median': np.median(group_scores),
                    'min': np.min(group_scores),
                    'max': np.max(group_scores)
                })
        
        group_stats_df = pd.DataFrame(group_stats)
        self.results['group_stats'] = group_stats_df
        return group_stats_df
    
    def plot_results(self, output_dir="./plots", top_n=20):
        """
        Generate visualization plots for the results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        top_n : int
            Number of top modules to show in detailed plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if 'anova' not in self.results:
            print("No ANOVA results to plot. Run calculate_module_anova() first.")
            return
        
        anova_results = self.results['anova']
        
        # 1. Effect size distribution
        plt.figure(figsize=(10, 6))
        plt.hist(anova_results['eta_squared'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Effect Size (η²)')
        plt.ylabel('Number of Modules')
        plt.title('Distribution of Effect Sizes (Group Association Strength)')
        plt.axvline(0.01, color='red', linestyle='--', label='Small effect (0.01)')
        plt.axvline(0.06, color='orange', linestyle='--', label='Medium effect (0.06)')
        plt.axvline(0.14, color='green', linestyle='--', label='Large effect (0.14)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'effect_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Volcano plot (effect size vs significance)
        plt.figure(figsize=(10, 8))
        x = anova_results['eta_squared']
        y = -np.log10(anova_results['p_value_corrected'].fillna(1))
        
        # Color points based on significance
        colors = ['red' if sig else 'blue' for sig in anova_results['significant'].fillna(False)]
        
        plt.scatter(x, y, c=colors, alpha=0.6)
        plt.xlabel('Effect Size (η²)')
        plt.ylabel('-log₁₀(Corrected p-value)')
        plt.title('Module Association Volcano Plot')
        
        # Add significance threshold line
        plt.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p = 0.05')
        plt.legend(['Significant', 'Non-significant', 'p = 0.05'])
        
        # Annotate top modules
        top_modules = anova_results.head(10)
        for _, row in top_modules.iterrows():
            if not pd.isna(row['eta_squared']) and not pd.isna(row['p_value_corrected']):
                plt.annotate(row['Module'], 
                           (row['eta_squared'], -np.log10(row['p_value_corrected'])),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'volcano_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Heatmap of top modules by group
        if 'group_stats' in self.results:
            top_modules = anova_results.head(top_n)['Module'].tolist()
            group_stats = self.results['group_stats']
            
            # Create pivot table for heatmap
            heatmap_data = group_stats[group_stats['Module'].isin(top_modules)].pivot(
                index='Module', columns='Group', values='mean'
            )
            
            plt.figure(figsize=(12, max(8, len(top_modules) * 0.4)))
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0)
            plt.title(f'Mean Module Usage Scores by Group (Top {top_n} Modules)')
            plt.tight_layout()
            plt.savefig(output_dir / f'top_{top_n}_modules_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"Plots saved to {output_dir}")
    
    def generate_report(self, output_file="module_group_analysis_report.txt"):
        """Generate a summary report of the analysis."""
        report = []
        report.append("=" * 60)
        report.append("MODULE-GROUP ASSOCIATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Data summary
        report.append(f"Data Summary:")
        report.append(f"- Number of modules: {len(self.usage_matrix)}")
        report.append(f"- Number of samples: {len(self.usage_matrix.columns)}")
        report.append(f"- Number of groups: {len(self.sample_metadata['Group'].unique())}")
        report.append(f"- Groups: {', '.join(self.sample_metadata['Group'].unique())}")
        report.append("")
        
        # Group sizes
        group_counts = self.sample_metadata['Group'].value_counts()
        report.append("Sample sizes per group:")
        for group, count in group_counts.items():
            report.append(f"- {group}: {count} samples")
        report.append("")
        
        # ANOVA results
        if 'anova' in self.results:
            anova_results = self.results['anova']
            n_significant = anova_results['significant'].sum()
            
            report.append(f"ANOVA Results:")
            report.append(f"- Modules tested: {len(anova_results)}")
            report.append(f"- Significant modules (FDR < 0.05): {n_significant}")
            report.append(f"- Percentage significant: {100 * n_significant / len(anova_results):.1f}%")
            report.append("")
            
            # Top modules
            top_modules = anova_results.head(10)
            report.append("Top 10 modules by effect size:")
            for _, row in top_modules.iterrows():
                sig_marker = "*" if row['significant'] else ""
                report.append(f"- {row['Module']}: η² = {row['eta_squared']:.3f}, "
                            f"p = {row['p_value_corrected']:.3e}{sig_marker}")
            report.append("")
        
        # PERMANOVA results
        if 'permanova' in self.results and self.results['permanova'] is not None:
            perm_results = self.results['permanova']
            report.append(f"PERMANOVA Results:")
            report.append(f"- Test statistic: {perm_results['test_statistic']:.4f}")
            report.append(f"- p-value: {perm_results['p_value']:.4f}")
            report.append(f"- Distance metric: {perm_results['distance_metric']}")
            report.append(f"- Permutations: {perm_results['n_permutations']}")
            if 'r2' in perm_results:
                report.append(f"- R^2 (variance explained): {perm_results['r2']:.4f}")
            if 'r2_adj' in perm_results and not np.isnan(perm_results['r2_adj']):
                report.append(f"- Adjusted R^2: {perm_results['r2_adj']:.4f}")
            report.append("")
            
            print(perm_results['p_value'])
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to {output_file}")
        print("\n" + report_text)
        
        return report_text


def load_example_data():
    """Generate example data for testing."""
    print("Generating example data...")
    
    # Create example usage matrix
    np.random.seed(42)
    n_modules = 50
    n_samples = 60
    
    # Create sample names
    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]
    module_names = [f"Module_{i:02d}" for i in range(n_modules)]
    
    # Create groups with some imbalance
    groups = (['Group_A'] * 20 + ['Group_B'] * 25 + ['Group_C'] * 15)
    
    # Create usage matrix with some group-specific patterns
    usage_matrix = np.random.normal(0, 1, (n_modules, n_samples))
    
    # Make some modules group-specific
    for i in range(10):  # First 10 modules are group-specific
        if i < 3:  # Group A specific
            usage_matrix[i, :20] += 2
        elif i < 6:  # Group B specific  
            usage_matrix[i, 20:45] += 2
        else:  # Group C specific
            usage_matrix[i, 45:] += 2
    
    usage_df = pd.DataFrame(usage_matrix, index=module_names, columns=sample_names)
    
    # Create metadata
    metadata_df = pd.DataFrame({
        'SampleName': sample_names,
        'Group': groups
    })
    
    return usage_df, metadata_df


def run_module_group_analysis(
    usage_df,
    metadata,
    #matrix_sep,
    output_dir,
    design_factor,
    sample_column,
    alpha: float = 0.05,
    example_data: bool = False):

    """
    Run the module-group association pipeline programmatically.

    Parameters
    ----------
    usage_matrix
        - str | Path: path to usage matrix file
        - pd.DataFrame: modules x samples or samples x modules (if transpose_matrix=True)
        - None: allowed only if example_data=True
    metadata
        - str | Path: path to metadata CSV
        - pd.DataFrame: must contain columns ['SampleName','Group']
        - None: allowed only if example_data=True
    transpose_matrix
        If True, transpose usage_matrix after loading (use when the file/DF has samples as rows).
    matrix_sep
        Separator for the usage matrix file (ignored if a DataFrame is provided).
    output_dir
        Where results will be written. Created if missing.
    alpha
        FDR threshold for ANOVA multiple testing correction.
    example_data
        If True, ignores `usage_matrix` and `metadata` and loads synthetic example data.

    Returns
    -------
    Dict[str, Any] with:
        - 'output_dir': Path
        - 'group_stats': pd.DataFrame
        - 'anova_results': pd.DataFrame
        - 'permanova_results': Optional[dict]
        - 'plots_dir': Path
        - 'report_path': Path
    """
    
    # ---- Load inputs ----
    if example_data:
        usage_df, meta_df = load_example_data()
    else:
        if usage_df is None or metadata is None:
            raise ValueError("Provide `usage_matrix` and `metadata`, or set example_data=True.")
        # usage matrix
        #if isinstance(usage_df, (str, Path)):
        #    usage_matrix = pd.read_csv(usage_matrix, index_col=0, sep=matrix_sep)
        if isinstance(usage_df, pd.DataFrame):
            usage_matrix = usage_df.copy()
        else:
            raise TypeError("usage_matrix must be str, Path, or pd.DataFrame.")
        # optional transpose to enforce modules x samples
        #if transpose_matrix:
            # usage_matrix = usage_matrix.set_index("SampleName")
            
            # usage_matrix = usage_matrix.reset_index()
            # usage_matrix = usage_matrix.T
            # usage_matrix = usage_matrix.rename_axis(None, axis=1).rename(columns=usage_matrix.iloc[0]).iloc[1:]
            # print("Matrix transposed: samples are now columns, modules are rows")

        # metadata
        if isinstance(metadata, (str, Path)):
            meta_df = pd.read_csv(metadata)
        elif isinstance(metadata, pd.DataFrame):
            meta_df = metadata.copy()
        else:
            raise TypeError("metadata must be str, Path, or pd.DataFrame.")
    
    meta_df = meta_df.rename(columns={design_factor: "Group", sample_column: "SampleName"})
    # ---- Output directory ----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Initialize analyzer (handles validation + alignment) ----
    analyzer = ModuleGroupAnalyzer(usage_matrix, meta_df)
    
    # Run analyses
    print("Starting analysis...")
    
    # Calculate group statistics
    #group_stats = analyzer.calculate_group_statistics()
    #group_stats.to_csv(output_dir / 'group_statistics.csv', index=False)
    
    # Perform ANOVA
    #anova_results = analyzer.calculate_module_anova(alpha=0.05)
    #anova_results.to_csv(output_dir / 'anova_results.csv', index=False)
    
    # Perform PERMANOVA
    #permanova_results = analyzer.perform_permanova()
    
   # if permanova_results is not None:
            #pd.DataFrame([permanova_results]).to_csv(output_dir / 'permanova_summary.tsv', sep='\t', index=False)
    silhouette_score = analyzer.calculate_silhouette_score()
    #plots_dir = output_dir / "plots"
    #analyzer.plot_results(plots_dir)

    #report_path = output_dir / "analysis_report.txt"
    #analyzer.generate_report(report_path)

    print(f"Analysis complete! Results saved to {output_dir}")

    return silhouette_score

    # return {
    #     "output_dir": output_dir,
    #     "group_stats": group_stats,
    #    "anova_results": anova_results,
    #     "permanova_results": permanova_results,
    #    "plots_dir": plots_dir,
    #    "report_path": report_path,
    # }