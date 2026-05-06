
import time
import pandas as pd
import yaml
from pathlib import Path
import numpy as np

# Assuming SymbolicRegressor is accessible
from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor

class DatasetLoader:
    """Loads a dataset from a directory with train/valid/id_test/ood_test.csv and metadata.yaml."""
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        self.metadata = self._load_metadata()
        self.data = self._load_csv_data()

    def _load_metadata(self):
        metadata_file = self.dataset_path / "metadata.yaml"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.yaml not found in {self.dataset_path}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_csv(self, filename: str):
        file_path = self.dataset_path / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found in {self.dataset_path}, returning empty data.")
            return pd.DataFrame()
        return pd.read_csv(file_path)

    def _load_csv_data(self):
        # Load all CSVs
        train_df = self._load_csv("train.csv")
        valid_df = self._load_csv("valid.csv")
        id_test_df = self._load_csv("id_test.csv")
        ood_test_df = self._load_csv("ood_test.csv")

        # Determine target column from metadata, or assume last column
        target_name = self.metadata.get('target', {}).get('name')
        if not target_name:
            print("Warning: Target column not specified in metadata.yaml. Attempting to infer from 'expression_str' or last column.")
            if 'expression_str' in self.metadata and not train_df.empty: # Common in BPG benchmarks
                 # Heuristic for BPG: if 'expression_str' is present, assume target is the last column
                target_name = train_df.columns[-1]
            elif not train_df.empty:
                target_name = train_df.columns[-1] # Fallback to last column
            
        if not target_name:
            if not train_df.empty:
                raise ValueError("Target column name must be specified in metadata.yaml or inferable from data.")
            else:
                print("Warning: No target column could be determined as training data is empty.")


        def split_X_y(df: pd.DataFrame):
            if df.empty:
                return np.array([]).reshape(0,len(df.columns) -1 if len(df.columns) > 0 else 0), np.array([]) # Handle empty dataframe for X
            if target_name not in df.columns:
                raise ValueError(f"Target column '{target_name}' not found in dataframe.")
            y = df[target_name].values
            X = df.drop(columns=[target_name]).values
            return X, y

        X_train, y_train = split_X_y(train_df)
        X_valid, y_valid = split_X_y(valid_df)
        X_id_test, y_id_test = split_X_y(id_test_df)
        X_ood_test, y_ood_test = split_X_y(ood_test_df)
        
        feature_names = [col for col in train_df.columns if col != target_name] if not train_df.empty else []

        return {
            "X_train": X_train, "y_train": y_train,
            "X_valid": X_valid, "y_valid": y_valid,
            "X_id_test": X_id_test, "y_id_test": y_id_test,
            "X_ood_test": X_ood_test, "y_ood_test": y_ood_test,
            "feature_names": feature_names
        }


class IterativeExperimentPipeline:
    """Manages an iterative symbolic regression experiment."""
    def __init__(self, dataset_dir: str, algorithm: str, params: dict = None, seed: int = 1314):
        self.dataset_dir = Path(dataset_dir)
        self.algorithm = algorithm
        self.params = params if params is not None else {}
        self.seed = seed
        self.dataset_loader = DatasetLoader(self.dataset_dir)
        
        # Initialize the SymbolicRegressor
        # It needs problem_name and experiments_dir for manifest generation
        self.regressor = SymbolicRegressor(
            tool_name=self.algorithm,
            problem_name=self.dataset_dir.name,
            seed=self.seed,
            **self.params
        )
        self.experiment_results = []
        self.total_fit_time = 0.0

    def _evaluate_model(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Evaluates the current model on given data."""
        if len(y_true) == 0:
            return {"rmse": np.nan, "r2": np.nan, "nmse": np.nan, "acc_0_1": np.nan}
            
        y_pred = self.regressor.predict(X)
        metrics = regression_metrics(y_true, y_pred, acc_threshold=0.1)
        return {
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "nmse": metrics["nmse"],
            "acc_0_1": metrics["acc_tau"],
        }

    def run(self, num_iterations: int = 1):
        """Runs the iterative experiment."""
        print(f"Starting iterative experiment for {self.algorithm} on {self.dataset_dir.name}")
        print(f"Initial parameters: {self.params}")
        
        X_train, y_train = self.dataset_loader.data["X_train"], self.dataset_loader.data["y_train"]
        X_valid, y_valid = self.dataset_loader.data["X_valid"], self.dataset_loader.data["y_valid"]

        for i in range(num_iterations):
            print(f"\n--- Iteration {i + 1}/{num_iterations} ---")
            
            # Step 2.1: Perform (incremental) fit and record time
            fit_start_time = time.time()
            self.regressor.fit(X_train, y_train)
            fit_duration = time.time() - fit_start_time
            self.total_fit_time += fit_duration
            
            print(f"Fit duration for this iteration: {fit_duration:.4f} seconds")
            print(f"Cumulative fit time: {self.total_fit_time:.4f} seconds")

            # Step 2.2: Validate
            valid_metrics = self._evaluate_model(X_valid, y_valid)
            print(f"Validation Metrics: RMSE={valid_metrics['rmse']:.4f}, R2={valid_metrics['r2']:.4f}")
            
            # Get optimal equation (if available)
            try:
                optimal_equation = self.regressor.get_optimal_equation()
            except Exception as e:
                optimal_equation = f"Error getting equation: {e}"
            print(f"Optimal Equation: {optimal_equation}")

            self.experiment_results.append({
                "iteration": i + 1,
                "fit_duration": fit_duration,
                "cumulative_fit_time": self.total_fit_time,
                "validation_metrics": valid_metrics,
                "optimal_equation": optimal_equation
            })

        print(f"\n--- Experiment Finished after {num_iterations} iterations ---")
        print(f"Total Cumulative Fit Time: {self.total_fit_time:.4f} seconds")

        # Step 3: Final evaluation on test sets
        print("\n--- Final Evaluation ---")
        id_test_metrics = self._evaluate_model(self.dataset_loader.data["X_id_test"], self.dataset_loader.data["y_id_test"])
        ood_test_metrics = self._evaluate_model(self.dataset_loader.data["X_ood_test"], self.dataset_loader.data["y_ood_test"])
        
        print(f"ID Test Metrics: RMSE={id_test_metrics['rmse']:.4f}, R2={id_test_metrics['r2']:.4f}")
        print(f"OOD Test Metrics: RMSE={ood_test_metrics['rmse']:.4f}, R2={ood_test_metrics['r2']:.4f}")

        final_report = {
            "dataset": self.dataset_dir.name,
            "algorithm": self.algorithm,
            "parameters": self.params,
            "seed": self.seed,
            "num_iterations": num_iterations,
            "total_fit_time": self.total_fit_time,
            "iteration_history": self.experiment_results,
            "final_id_test_metrics": id_test_metrics,
            "final_ood_test_metrics": ood_test_metrics,
            "final_optimal_equation": optimal_equation # Last one found
        }
        
        return final_report

# Example usage (for testing purposes, not part of the module's main functionality)
if __name__ == "__main__":
    # Create a dummy dataset directory for testing
    dummy_dataset_dir = Path("dummy_bpg_dataset")
    dummy_dataset_dir.mkdir(exist_ok=True)

    # Create dummy metadata.yaml
    with open(dummy_dataset_dir / "metadata.yaml", "w") as f:
        f.write("""
target: {name: y}
expression_str: \"x0**2 + x1\"
        """)
    
    # Create dummy CSVs
    np.random.seed(0)
    X_dummy = np.random.rand(100, 2) * 5
    y_dummy = X_dummy[:, 0]**2 + X_dummy[:, 1] + 0.1 * np.random.randn(100)

    df_train = pd.DataFrame(X_dummy[:80], columns=['x0', 'x1'])
    df_train['y'] = y_dummy[:80]
    df_train.to_csv(dummy_dataset_dir / "train.csv", index=False)

    df_valid = pd.DataFrame(X_dummy[80:90], columns=['x0', 'x1'])
    df_valid['y'] = y_dummy[80:90]
    df_valid.to_csv(dummy_dataset_dir / "valid.csv", index=False)

    df_id_test = pd.DataFrame(X_dummy[90:], columns=['x0', 'x1'])
    df_id_test['y'] = y_dummy[90:]
    df_id_test.to_csv(dummy_dataset_dir / "id_test.csv", index=False)

    # For OOD, let's make some values outside the training range
    X_ood_dummy = np.random.rand(10, 2) * 10 + 5 # Larger range
    y_ood_dummy = X_ood_dummy[:, 0]**2 + X_ood_dummy[:, 1] + 0.1 * np.random.randn(10)
    df_ood_test = pd.DataFrame(X_ood_dummy, columns=['x0', 'x1'])
    df_ood_test['y'] = y_ood_dummy
    df_ood_test.to_csv(dummy_dataset_dir / "ood_test.csv", index=False)


    # Example of running the pipeline (requires a working 'gplearn' setup in conda)
    try:
        print("\n--- Running dummy experiment with gplearn (requires 'sim' conda env setup) ---")
        pipeline = IterativeExperimentPipeline(
            dataset_dir=str(dummy_dataset_dir),
            algorithm="gplearn", # or "pysr", "llmsr" if configured
            params={"population_size": 100, "generations": 5, "random_state": 42},
            seed=42
        )
        report = pipeline.run(num_iterations=2) # Run 2 iterations of fit
        print("\nFinal Report:")
        # For simplicity, print just a summary
        print(f"Dataset: {report['dataset']}")
        print(f"Algorithm: {report['algorithm']}")
        print(f"Total Fit Time: {report['total_fit_time']:.4f}s")
        print(f"Final ID Test RMSE: {report['final_id_test_metrics']['rmse']:.4f}")
        print(f"Final OOD Test RMSE: {report['final_ood_test_metrics']['rmse']:.4f}")
        print(f"Final Equation: {report['final_optimal_equation']}")

    except Exception as e:
        print(f"\nCould not run dummy experiment. Ensure 'gplearn' is set up in 'sim' conda environment and other dependencies are met.")
        print(f"Error: {e}")

    finally:
        # Clean up dummy dataset
        import shutil
        if dummy_dataset_dir.exists():
            shutil.rmtree(dummy_dataset_dir)
            print(f"\nCleaned up dummy dataset directory: {dummy_dataset_dir}")
