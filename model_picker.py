import time
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import get_scorer_names
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from typing import Dict, List, Tuple, Optional, Union

class ModelPicker:
    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        """
        Initialize the model selection class with comprehensive regression models.
        
        Parameters:
        - n_jobs: Number of parallel jobs to run (-1 uses all cores)
        - random_state: Random seed for reproducibility
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._initialize_models()
        self._initialize_metrics()
        self.results_df = pd.DataFrame()
        self.best_model = None
        self.cv_results = []

    def _initialize_models(self) -> None:
        """Initialize models with their default parameters and search grids."""
        self.models = [
            ('LinearRegression', LinearRegression(), {}),
            ('Ridge', Ridge(), {'alpha': np.logspace(-3, 3, 7)}),
            ('Lasso', Lasso(), {'alpha': np.logspace(-4, 0, 5)}),
            ('ElasticNet', ElasticNet(), {
                'alpha': np.logspace(-3, 1, 5),
                'l1_ratio': np.linspace(0.1, 0.9, 5)
            }),
            ('DecisionTree', DecisionTreeRegressor(random_state=self.random_state), {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }),
            ('RandomForest', RandomForestRegressor(random_state=self.random_state), {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            }),
            ('GradientBoosting', GradientBoostingRegressor(random_state=self.random_state), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }),
            ('SVR', SVR(), {
                'C': np.logspace(-1, 2, 4),
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }),
            ('KNN', KNeighborsRegressor(), {
                'n_neighbors': range(3, 8),
                'weights': ['uniform', 'distance']
            }),
            ('AdaBoost', AdaBoostRegressor(random_state=self.random_state), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            }),
            ('XGBoost', XGBRegressor(random_state=self.random_state), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            })
        ]

    def _initialize_metrics(self) -> None:
        """Initialize evaluation metrics."""
        self.metrics = {
            'R2': 'r2',
            'MAE': 'neg_mean_absolute_error',
            'RMSE': 'neg_root_mean_squared_error',
            'ExplainedVariance': 'explained_variance',
            'MaxError': 'max_error'
        }

    def _evaluate_model(self, model_name: str, model: RegressorMixin, 
                       params: Dict, X: np.ndarray, y: np.ndarray) -> Tuple[str, RegressorMixin, Dict]:
        """
        Perform model evaluation with hyperparameter tuning and cross-validation.
        
        Returns:
        - Tuple of (model_name, best_model, evaluation_results)
        """
        try:
            start_time = time.time()
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Update parameter names for pipeline
            pipe_params = {f'model__{k}': v for k, v in params.items()}
            
            # Grid search with inner CV
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=pipe_params,
                cv=KFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='neg_root_mean_squared_error',
                n_jobs=self.n_jobs,
                verbose=0
            )
            grid.fit(X, y)
            
            # Outer CV evaluation with best params
            cv_results = cross_validate(
                grid.best_estimator_,
                X, y,
                cv=KFold(n_splits=10, shuffle=True, random_state=self.random_state),
                scoring=self.metrics,
                n_jobs=self.n_jobs,
                return_train_score=False
            )
            
            elapsed_time = time.time() - start_time
            
            # Store results
            results = {
                'Model': model_name,
                'R2_mean': np.mean(cv_results['test_R2']),
                'R2_std': np.std(cv_results['test_R2']),
                'MAE_mean': -np.mean(cv_results['test_MAE']),
                'RMSE_mean': -np.mean(cv_results['test_RMSE']),
                'ExplVar_mean': np.mean(cv_results['test_ExplainedVariance']),
                'MaxError_mean': np.mean(cv_results['test_MaxError']),
                'Time': elapsed_time,
                'Best_Params': grid.best_params_
            }
            
            print(f"{model_name} completed in {elapsed_time:.2f}s (Best R2: {results['R2_mean']:.3f})")
            
            return model_name, grid.best_estimator_, results
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return model_name, None, None

    def pick(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series]) -> Optional[RegressorMixin]:
        """
        Evaluate all models and select the best performing one.
        
        Parameters:
        - X: Feature matrix
        - y: Target vector
        
        Returns:
        - Best trained model or None if all failed
        """
        self.results = []
        successful_models = 0
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.models), 4)) as executor:
            futures = {
                executor.submit(self._evaluate_model, name, model, params, X, y): name
                for name, model, params in self.models
            }
            
            for future in concurrent.futures.as_completed(futures):
                model_name, model, results = future.result()
                if model is not None and results is not None:
                    self.results.append(results)
                    successful_models += 1
        
        if successful_models == 0:
            print("Warning: All model evaluations failed!")
            return None
            
        # Create results dataframe
        self.results_df = pd.DataFrame(self.results).sort_values('R2_mean', ascending=False)
        self.best_model = self.models[[m[0] for m in self.models].index(self.results_df.iloc[0]['Model'])][1]
        
        print("\n=== Model Comparison Results ===")
        print(self.results_df.to_string(index=False))
        
        return self.best_model

    def plot(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        """Generate comprehensive model comparison visualizations."""
        if self.results_df.empty:
            print("No results to plot. Run pick() first.")
            return
            
        plt.figure(figsize=figsize)
        
        # 1. R2 Score Comparison
        plt.subplot(2, 2, 1)
        sns.barplot(
            x='R2_mean', y='Model', 
            data=self.results_df.sort_values('R2_mean', ascending=True),
            # xerr=self.results_df['R2_std'],
            palette='viridis'
        )
        plt.title('R² Score Comparison (Higher Better)\nMean ± 1 SD', pad=20)
        plt.xlabel('R² Score')
        plt.ylabel('')
        
        # 2. Error Metrics Comparison
        plt.subplot(2, 2, 2)
        error_metrics = self.results_df.melt(
            id_vars='Model',
            value_vars=['MAE_mean', 'RMSE_mean', 'MaxError_mean'],
            var_name='Metric',
            value_name='Value'
        )
        error_metrics['Metric'] = error_metrics['Metric'].str.replace('_mean', '')
        sns.barplot(
            x='Value', y='Model', hue='Metric',
            data=error_metrics,
            palette='coolwarm'
        )
        plt.title('Error Metrics Comparison (Lower Better)', pad=20)
        plt.xlabel('Error Value')
        plt.ylabel('')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Training Time Comparison
        plt.subplot(2, 2, 3)
        sns.barplot(
            x='Time', y='Model', 
            data=self.results_df.sort_values('Time', ascending=True),
            palette='magma'
        )
        plt.title('Training Time Comparison', pad=20)
        plt.xlabel('Time (seconds)')
        plt.ylabel('')
        
        # 4. Metric Distributions
        plt.subplot(2, 2, 4)
        metrics = ['R2', 'MAE', 'RMSE']
        for i, metric in enumerate(metrics):
            plt.scatter(
                self.results_df[f'{metric}_mean'], 
                self.results_df.index,
                label=metric,
                s=100,
                alpha=0.7
            )
        plt.title('Metric Trade-offs', pad=20)
        plt.xlabel('Metric Value')
        plt.yticks(range(len(self.results_df)), self.results_df['Model'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def get_best_params(self) -> Dict:
        """Return the best parameters found during model selection."""
        if not self.results_df.empty:
            best_model_name = self.results_df.iloc[0]['Model']
            best_params = self.results_df[self.results_df['Model'] == best_model_name]['Best_Params'].iloc[0]
            return {best_model_name: best_params}
        return {}

    def get_results(self) -> pd.DataFrame:
        """Return the complete evaluation results as a DataFrame."""
        return self.results_df.copy()