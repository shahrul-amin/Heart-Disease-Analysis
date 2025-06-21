"""
Visualization Utilities for Heart Disease Data Analysis
Interactive and static visualizations for data exploration and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class HeartDiseaseVisualizer:
    """
    Comprehensive visualization toolkit for heart disease data analysis
    """    
    def __init__(self):
        self.color_palette = ['#e50914', '#f40612', '#831010', '#b20710', '#ff0a16', '#ffffff', '#cccccc', '#999999']
        self.theme_colors = {
            'primary': '#e50914',
            'secondary': '#f40612',
            'accent': '#ffffff',
            'success': '#46d369',
            'background': '#000000',
            'text': '#ffffff',
            'grid': '#333333',            'card_bg': '#141414'
        }
    
    def set_plot_style(self):
        """Set consistent plot styling for Netflix theme"""
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#000000'
        plt.rcParams['axes.facecolor'] = '#141414'
        plt.rcParams['text.color'] = '#ffffff'
        plt.rcParams['axes.labelcolor'] = '#ffffff'
        plt.rcParams['xtick.color'] = '#ffffff'
        plt.rcParams['ytick.color'] = '#ffffff'
        plt.rcParams['grid.color'] = '#333333'
        plt.rcParams['axes.edgecolor'] = '#333333'
        sns.set_palette(self.color_palette)
        
    def create_overview_dashboard(self, df):
        """Create comprehensive overview dashboard"""
        # Calculate key metrics
        total_patients = len(df)
        heart_disease_count = df['num'].sum() if 'num' in df.columns else 0
        avg_age = df['age'].mean() if 'age' in df.columns else 0
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Age Distribution', 'Heart Disease by Gender', 
                'Chest Pain Types', 'Blood Pressure Distribution',
                'Cholesterol Levels', 'Heart Disease Prevalence'
            ],
            specs=[[{"type": "histogram"}, {"type": "bar"}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Age Distribution
        fig.add_trace(
            go.Histogram(x=df['age'], name='Age', nbinsx=20, 
                        marker_color=self.theme_colors['primary']),
            row=1, col=1
        )
        
        # Heart Disease by Gender
        if 'sex' in df.columns:
            gender_disease = df.groupby(['sex', 'num']).size().unstack(fill_value=0)
            for i, disease_status in enumerate(gender_disease.columns):
                fig.add_trace(
                    go.Bar(x=gender_disease.index, y=gender_disease[disease_status],
                          name=f'Disease Status {disease_status}',
                          marker_color=self.color_palette[i]),
                    row=1, col=2
                )
        
        # Chest Pain Types
        if 'cp' in df.columns:
            cp_counts = df['cp'].value_counts()
            fig.add_trace(
                go.Pie(labels=cp_counts.index, values=cp_counts.values,
                      name="Chest Pain"),
                row=1, col=3
            )
        
        # Blood Pressure Distribution
        if 'trestbps' in df.columns:
            fig.add_trace(
                go.Box(y=df['trestbps'], name='Blood Pressure',
                      marker_color=self.theme_colors['secondary']),
                row=2, col=1
            )
        
        # Cholesterol Levels
        if 'chol' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['chol'], name='Cholesterol', nbinsx=25,
                           marker_color=self.theme_colors['accent']),
                row=2, col=2
            )
        
        # Heart Disease Prevalence
        disease_counts = df['num'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=disease_counts.index, y=disease_counts.values,
                  name='Heart Disease Severity',
                  marker_color=self.theme_colors['success']),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            title_text="Heart Disease Dataset - Comprehensive Overview",
            title_x=0.5,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, df, figsize=(12, 10)):
        """Create interactive correlation heatmap"""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            title_x=0.5,
            width=800,
            height=600
        )
        
        return fig
    
    def create_distribution_plots(self, df, columns=None):
        """Create distribution plots for specified columns"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns[:6]
        
        n_cols = 2
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(columns):
            if i < len(axes):
                # Distribution plot
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_categorical_analysis(self, df, target_col='num'):
        """Create comprehensive categorical variable analysis"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return None
        
        n_cols = 2
        n_rows = (len(categorical_cols) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                # Create crosstab
                if target_col in df.columns:
                    crosstab = pd.crosstab(df[col], df[target_col])
                    crosstab.plot(kind='bar', ax=axes[i], rot=45)
                    axes[i].set_title(f'{col} vs {target_col}')
                else:
                    df[col].value_counts().plot(kind='bar', ax=axes[i], rot=45)
                    axes[i].set_title(f'Distribution of {col}')
                
                axes[i].legend(title=target_col if target_col in df.columns else '')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_outlier_analysis(self, df, columns=None):
        """Create outlier analysis visualization"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns[:6]
        
        # Create interactive box plots
        fig = make_subplots(
            rows=1, cols=len(columns),
            subplot_titles=columns
        )
        
        for i, col in enumerate(columns):
            fig.add_trace(
                go.Box(y=df[col], name=col, boxpoints='outliers'),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Outlier Analysis - Box Plots',
            title_x=0.5,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_bivariate_analysis(self, df, x_col, y_col, hue_col=None):
        """Create bivariate analysis plot"""
        fig = px.scatter(
            df, x=x_col, y=y_col, color=hue_col,
            title=f'{y_col} vs {x_col}',
            color_continuous_scale='viridis'
        )
        
        # Add trendline
        fig.add_trace(
            go.Scatter(
                x=df[x_col], 
                y=np.poly1d(np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1))(df[x_col]),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_feature_importance_plot(self, feature_names, importance_scores):
        """Create feature importance visualization"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Feature Importance Analysis',
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=max(400, len(feature_names) * 25))
        return fig
    
    def create_missing_values_plot(self, df):
        """Create missing values analysis plot"""
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        if len(missing_data) == 0:
            return None
        
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title='Missing Values by Column',
            labels={'x': 'Number of Missing Values', 'y': 'Columns'}
        )
        
        fig.update_layout(height=max(300, len(missing_data) * 30))
        return fig
    
    def create_preprocessing_comparison(self, original_df, processed_df):
        """Create before/after preprocessing comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Missing Values - Before', 'Missing Values - After',
                'Data Types - Before', 'Data Types - After'
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "domain"}, {"type": "domain"}]]
        )
          # Missing values comparison
        missing_orig = original_df.isnull().sum()
        missing_proc = processed_df.isnull().sum()
        
        # Convert index to strings to ensure JSON serialization
        fig.add_trace(
            go.Bar(x=[str(col) for col in missing_orig.index], y=missing_orig.values, 
                  name='Original', marker_color='red'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=[str(col) for col in missing_proc.index], y=missing_proc.values, 
                  name='Processed', marker_color='green'),
            row=1, col=2
        )
          # Data types comparison
        types_orig = original_df.dtypes.value_counts()
        types_proc = processed_df.dtypes.value_counts()
        
        # Convert dtype objects to strings for JSON serialization
        orig_labels = [str(dtype) for dtype in types_orig.index]
        proc_labels = [str(dtype) for dtype in types_proc.index]
        
        fig.add_trace(
            go.Pie(labels=orig_labels, values=types_orig.values, name="Original"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Pie(labels=proc_labels, values=types_proc.values, name="Processed"),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Preprocessing Impact Comparison',
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_target_analysis(self, df, target_col='num'):
        """Create comprehensive target variable analysis"""
        if target_col not in df.columns:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Target Distribution', 'Target by Age Groups',
                'Target by Gender', 'Target Statistics'
            ],
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Target distribution
        target_counts = df[target_col].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=target_counts.index, y=target_counts.values,
                  name='Target Distribution'),
            row=1, col=1
        )
        
        # Target by age groups
        if 'age' in df.columns:
            df_temp = df.copy()
            df_temp['age_group'] = pd.cut(df_temp['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Old', 'Very Old'])
            fig.add_trace(
                go.Box(x=df_temp['age_group'], y=df_temp[target_col],
                      name='Target by Age'),
                row=1, col=2
            )
        
        # Target by gender
        if 'sex' in df.columns:
            gender_target = df.groupby('sex')[target_col].mean()
            fig.add_trace(
                go.Bar(x=gender_target.index, y=gender_target.values,
                      name='Target by Gender'),
                row=2, col=1
            )
        
        # Target statistics table
        target_stats = df[target_col].describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[target_stats.index, target_stats.values.round(2)])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Target Variable ({target_col}) Analysis',
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_data_quality_dashboard(self, quality_report):
        """Create data quality assessment dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Quality Score Components', 'Missing Values by Column',
                'Data Quality Grade', 'Recommendations by Priority'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "pie"}]]
        )
        
        # Quality score components
        if 'overall_score' in quality_report:
            scores = quality_report['overall_score']['component_scores']
            fig.add_trace(
                go.Bar(x=list(scores.keys()), y=list(scores.values()),
                      name='Quality Scores'),
                row=1, col=1
            )
        
        # Missing values
        if 'completeness' in quality_report:
            missing_data = quality_report['completeness']['columns']
            cols = list(missing_data.keys())
            missing_rates = [missing_data[col]['missing_rate'] for col in cols]
            
            fig.add_trace(
                go.Bar(x=cols, y=missing_rates, name='Missing %'),
                row=1, col=2
            )
        
        # Overall quality indicator
        if 'overall_score' in quality_report:
            overall_score = quality_report['overall_score']['overall']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Quality Score"},
                    gauge={'axis': {'range': [None, 100]},
                          'bar': {'color': "darkblue"},
                          'steps': [
                              {'range': [0, 50], 'color': "lightgray"},
                              {'range': [50, 80], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 90}}
                ),
                row=2, col=1
            )
        
        # Recommendations by priority
        if 'recommendations' in quality_report:
            priority_counts = {}
            for rec in quality_report['recommendations']:
                priority = rec['priority']
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            fig.add_trace(
                go.Pie(labels=list(priority_counts.keys()), 
                      values=list(priority_counts.values()),
                      name="Recommendations"),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Data Quality Assessment Dashboard',
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_filter_dashboard(self, df):
        """Create fully interactive dashboard with real-time filtering"""
        # Create multi-level filtering interface
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age_range = st.slider(
                "Age Range", 
                int(df['age'].min()), 
                int(df['age'].max()), 
                (int(df['age'].min()), int(df['age'].max())),
                key="age_filter"
            )
        
        with col2:
            gender_filter = st.multiselect(
                "Gender", 
                df['sex'].unique() if 'sex' in df.columns else [],
                default=df['sex'].unique() if 'sex' in df.columns else [],
                key="gender_filter"
            )
        
        with col3:
            cp_filter = st.multiselect(
                "Chest Pain Type",
                df['cp'].unique() if 'cp' in df.columns else [],
                default=df['cp'].unique() if 'cp' in df.columns else [],
                key="cp_filter"
            )
        
        with col4:
            chol_range = st.slider(
                "Cholesterol Range",
                int(df['chol'].min()) if 'chol' in df.columns else 0,
                int(df['chol'].max()) if 'chol' in df.columns else 400,
                (int(df['chol'].min()) if 'chol' in df.columns else 0, 
                 int(df['chol'].max()) if 'chol' in df.columns else 400),
                key="chol_filter"
            )
        
        # Apply filters
        filtered_df = df[
            (df['age'] >= age_range[0]) & 
            (df['age'] <= age_range[1])
        ]
        
        if 'sex' in df.columns and gender_filter:
            filtered_df = filtered_df[filtered_df['sex'].isin(gender_filter)]
        
        if 'cp' in df.columns and cp_filter:
            filtered_df = filtered_df[filtered_df['cp'].isin(cp_filter)]
        
        if 'chol' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['chol'] >= chol_range[0]) & 
                (filtered_df['chol'] <= chol_range[1])
            ]
        
        # Display filter results
        st.info(f"Showing {len(filtered_df)} of {len(df)} records after filtering")
          # Create dynamic visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive scatter plot with NaN handling
            plot_df = filtered_df.copy()
            
            # Handle NaN values for size column
            size_col = None
            if 'thalch' in plot_df.columns:
                # Fill NaN values in size column
                plot_df['thalch'] = plot_df['thalch'].fillna(plot_df['thalch'].median())
                size_col = 'thalch'
            
            fig_scatter = px.scatter(
                plot_df,
                x='age' if 'age' in plot_df.columns else plot_df.columns[0],
                y='chol' if 'chol' in plot_df.columns else plot_df.columns[1],
                color='num' if 'num' in plot_df.columns else None,
                size=size_col,
                hover_data=['trestbps', 'oldpeak'] if all(col in plot_df.columns for col in ['trestbps', 'oldpeak']) else None,
                title="Interactive Patient Analysis",
                color_continuous_scale="Viridis"
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Dynamic correlation heatmap
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Dynamic Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        return filtered_df
    
    def create_advanced_analytics_dashboard(self, df):
        """Create advanced analytics with ML insights"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Prepare data for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # K-means clustering
        n_clusters = st.slider("Number of Clusters", 2, 8, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create PCA visualization
        fig_pca = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=clusters.astype(str),
            title=f"Patient Clustering (PCA Visualization) - {n_clusters} Clusters",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                    'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Cluster analysis
        cluster_df = df.copy()
        cluster_df['Cluster'] = clusters
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Characteristics")
            cluster_stats = cluster_df.groupby('Cluster')[numeric_cols].mean().round(2)
            st.dataframe(cluster_stats, use_container_width=True)
        
        with col2:
            # Feature importance for clustering
            st.subheader("Feature Importance in Clustering")
            feature_importance = pd.DataFrame({
                'Feature': numeric_cols,
                'PCA_Component_1': np.abs(pca.components_[0]),
                'PCA_Component_2': np.abs(pca.components_[1])
            }).sort_values('PCA_Component_1', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10),
                x='PCA_Component_1',
                y='Feature',
                orientation='h',
                title="Top Features Contributing to Clustering"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        return cluster_df
    
    def create_predictive_analytics_dashboard(self, df):
        """Create predictive analytics dashboard"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import shap
        
        if 'num' not in df.columns:
            st.warning("Target variable 'num' not found for predictive analysis")
            return
        
        # Prepare data
        X = df.drop('num', axis=1).select_dtypes(include=[np.number])
        y = df['num']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_importance = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance for Heart Disease Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Prediction distribution
            pred_dist = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            fig_pred = px.histogram(
                pred_dist,
                x='Predicted',
                color='Actual',
                title="Prediction Distribution by Actual Values",
                barmode='group'
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        # Interactive prediction
        st.subheader("🔮 Interactive Prediction")
        
        prediction_cols = st.columns(len(X.columns[:6]))  # Limit to first 6 features
        
        user_input = {}
        for i, col in enumerate(X.columns[:6]):
            with prediction_cols[i]:
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                mean_val = float(X[col].mean())
                user_input[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"pred_{col}"
                )
        
        if st.button("🎯 Predict Heart Disease Risk"):
            # Create prediction input
            pred_input = pd.DataFrame([user_input])
            
            # Add missing columns with mean values
            for col in X.columns:
                if col not in pred_input.columns:
                    pred_input[col] = X[col].mean()
            
            # Reorder columns to match training data
            pred_input = pred_input[X.columns]
            
            # Make prediction
            prediction = model.predict(pred_input)[0]
            probability = model.predict_proba(pred_input)[0]
            
            if prediction == 0:
                st.success(f"✅ Low Risk: {probability[0]:.2%} chance of no heart disease")
            else:
                st.error(f"⚠️ High Risk: {probability[1] if len(probability) > 1 else probability[0]:.2%} chance of heart disease")