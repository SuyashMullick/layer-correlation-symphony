import os
import subprocess
import time
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
import platform

app = Flask(__name__)

# Cross-platform setup
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXEC = Path(sys.executable)
UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
OUTPUT_FOLDER = PROJECT_ROOT / 'static' / 'results'
DATA_FOLDER = PROJECT_ROOT / 'data' / 'aligned'

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER


def get_available_tif_files():
    """Scan the data folder and return all .tif files with relative paths"""
    if not DATA_FOLDER.exists():
        return []
    
    tif_files = []
    for tif_path in DATA_FOLDER.rglob('*.tif'):
        # Use forward slashes for consistency across platforms
        relative_path = tif_path.relative_to(PROJECT_ROOT)
        tif_files.append(str(relative_path).replace('\\', '/'))
    
    return sorted(tif_files)


def _check_python():
    if not PYTHON_EXEC.exists():
        return {'status': 'FAILURE', 'error': f"Python executable not found at {PYTHON_EXEC}. Ensure your environment is activated."}
    return None


def run_comparison_script(filepath_a: str, filepath_b: str, output_id: str) -> dict:
    python_check = _check_python()
    if python_check: return python_check

    run_output_dir = OUTPUT_FOLDER / output_id
    run_output_dir.mkdir(exist_ok=True)

    # Convert paths to absolute and use os-specific format
    abs_path_a = str(Path(filepath_a).resolve())
    abs_path_b = str(Path(filepath_b).resolve())
    abs_output = str(run_output_dir.resolve())

    command = [
        str(PYTHON_EXEC), '-m', 'cli.symph-compare',
        '--a', abs_path_a,
        '--b', abs_path_b,
        '--out', abs_output,
        '--nodata', '-9999'
    ]

    # Print debug command (cross-platform friendly)
    print(f"\n[DEBUG] Running command:")
    print(f"  Python: {PYTHON_EXEC}")
    print(f"  Working Dir: {PROJECT_ROOT}")
    print(f"  Command: {' '.join(command)}\n")

    try:
        result = subprocess.run(
            command, 
            cwd=str(PROJECT_ROOT), 
            capture_output=True, 
            text=True, 
            check=True,
            env=os.environ.copy()  # Pass environment variables
        )
        
        metrics_path = run_output_dir / 'metrics.csv'

        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            stats = metrics_df.iloc[0].to_dict()
        else:
            stats = {'error': 'Metrics file not generated. Check stdout/stderr.'}

        return {
            'status': 'SUCCESS', 'output_id': output_id, 'metrics': stats,
            'stdout': result.stdout, 'stderr': result.stderr
        }

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subprocess failed:\n{e.stderr}")
        return {
            'status': 'FAILURE', 'output_id': output_id,
            'error': f"Compare Script execution failed. Error: {e.stderr}",
            'stderr': e.stderr, 'stdout': e.stdout
        }
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        return {'status': 'FAILURE', 'error': f"General execution error: {e}"}


def run_prediction_script(target_file: str, predictor_files: list, model: str, output_id: str) -> dict:
    python_check = _check_python()
    if python_check: return python_check

    run_output_dir = OUTPUT_FOLDER / output_id
    run_output_dir.mkdir(exist_ok=True)

    # Convert all paths to absolute
    abs_target = str(Path(target_file).resolve())
    abs_predictors = [str(Path(p).resolve()) for p in predictor_files]
    abs_output = str(run_output_dir.resolve())

    command = [
        str(PYTHON_EXEC), '-m', 'cli.symph-predict',
        '--target', abs_target,
        '--predictors'
    ]
    command.extend(abs_predictors)
    command.extend([
        '--out', abs_output,
        '--model', model,
        '--transform_y', 'log1p',
        '--sample', '200000'
    ])

    print(f"\n[DEBUG] Running prediction command:")
    print(f"  Command: {' '.join(command)}")
    print(f"  Python Executable: {PYTHON_EXEC}")
    print(f"  Working Directory: {PROJECT_ROOT}")
    print(f"  Output Directory: {abs_output}")
    print(f"  Target File: {abs_target}")
    try:
        stat = os.stat(abs_target)
        print(f"    Target File Size: {stat.st_size} bytes, Modified: {time.ctime(stat.st_mtime)}")
    except Exception as e:
        print(f"    Could not stat target file: {e}")
    print(f"  Predictor Files:")
    for p in abs_predictors:
        try:
            stat = os.stat(p)
            print(f"    {p} | Size: {stat.st_size} bytes, Modified: {time.ctime(stat.st_mtime)}")
        except Exception as e:
            print(f"    {p} | Could not stat file: {e}")
    """
    print(f"  Environment Variables (partial):")
    for k in ['PATH', 'PYTHONPATH', 'GDAL_DATA', 'PROJ_LIB']:
        print(f"    {k}: {os.environ.get(k, '[not set]')}")
    print()
    """
    try:
        result = subprocess.run(
            command, 
            cwd=str(PROJECT_ROOT), 
            capture_output=True, 
            text=True, 
            check=True,
            env=os.environ.copy()
        )
        
        metrics_path = run_output_dir / 'metrics.csv'

        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            stats = metrics_df.iloc[0].to_dict()
        else:
            stats = {'error': 'Metrics file not generated.'}

        return {
            'status': 'SUCCESS', 'output_id': output_id, 'metrics': stats,
            'stdout': result.stdout, 'stderr': result.stderr
        }

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subprocess failed:\n{e.stderr}")
        return {
            'status': 'FAILURE', 'output_id': output_id,
            'error': f"Predict Script execution failed. Error: {e.stderr}",
            'stderr': e.stderr, 'stdout': e.stdout
        }
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        return {'status': 'FAILURE', 'error': f"General execution error: {e}"}


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare')
def compare_page():
    available_files = get_available_tif_files()
    return render_template('compare.html', available_files=available_files)


@app.route('/predict')
def predict_page():
    model_choices = ['rf', 'gbm', 'xgb', 'nn', 'auto']
    available_files = get_available_tif_files()
    return render_template('predict.html', model_choices=model_choices, available_files=available_files)


@app.route('/run_compare', methods=['POST'])
def run_compare():
    # Check if using file paths or uploaded files
    file_path_1 = request.form.get('file_path_1', '').strip()
    file_path_2 = request.form.get('file_path_2', '').strip()
    
    uploaded_file1 = request.files.get('user_file1')
    uploaded_file2 = request.files.get('user_file2')
    
    unique_id = str(int(time.time()))
    temp_files = []
    
    try:
        # Determine filepath for file 1
        if uploaded_file1 and uploaded_file1.filename:
            # User uploaded a file
            temp_filename1 = f"{unique_id}_A_{secure_filename(uploaded_file1.filename)}"
            filepath1 = UPLOAD_FOLDER / temp_filename1
            uploaded_file1.save(str(filepath1))
            temp_files.append(filepath1)
        elif file_path_1:
            # User selected existing file - convert to proper Path object
            filepath1 = PROJECT_ROOT / file_path_1.replace('/', os.sep)
            if not filepath1.exists():
                return f"Error: Selected file does not exist: {file_path_1}", 400
        else:
            return "Error: Must provide file 1 (either select or upload)", 400
        
        # Determine filepath for file 2
        if uploaded_file2 and uploaded_file2.filename:
            temp_filename2 = f"{unique_id}_B_{secure_filename(uploaded_file2.filename)}"
            filepath2 = UPLOAD_FOLDER / temp_filename2
            uploaded_file2.save(str(filepath2))
            temp_files.append(filepath2)
        elif file_path_2:
            filepath2 = PROJECT_ROOT / file_path_2.replace('/', os.sep)
            if not filepath2.exists():
                return f"Error: Selected file does not exist: {file_path_2}", 400
        else:
            return "Error: Must provide file 2 (either select or upload)", 400
        
        # Run the comparison (pass as strings)
        result_data = run_comparison_script(str(filepath1), str(filepath2), unique_id)
        
        if result_data['status'] == 'SUCCESS':
            return redirect(url_for('results_page', run_id=unique_id))
        else:
            return render_template('error.html', data=result_data), 500

    finally:
        # Clean up uploaded temporary files only
        for temp_file in temp_files:
            try: 
                os.remove(str(temp_file))
            except Exception as e:
                print(f"[WARNING] Could not delete temp file {temp_file}: {e}")


@app.route('/run_predict', methods=['POST'])
def run_predict():
    # Accept both file uploads and file selections
    target_path = request.form.get('target_path', '').strip()
    predictor_paths = request.form.getlist('predictor_paths')
    target_file = request.files.get('target_file')
    predictor_files = request.files.getlist('predictor_files')
    model_choice = request.form.get('model_choice')

    unique_id = str(int(time.time()))
    temp_files = []

    # Determine target file path
    if target_file and target_file.filename:
        temp_filename = f"{unique_id}_T_{secure_filename(target_file.filename)}"
        target_filepath = UPLOAD_FOLDER / temp_filename
        target_file.save(str(target_filepath))
        temp_files.append(target_filepath)
        target_path_final = str(target_filepath)
    elif target_path:
        # Use selected file from disk
        target_path_final = str(PROJECT_ROOT / target_path.replace('/', os.sep))
        if not Path(target_path_final).exists():
            return f"Error: Target file does not exist: {target_path}", 400
    else:
        return "Error: Must provide a target file", 400

    # Collect predictor file paths
    predictor_filepaths = []
    # Uploaded files
    for i, file in enumerate(predictor_files):
        if file and file.filename:
            temp_filename = f"{unique_id}_P{i}_{secure_filename(file.filename)}"
            pred_filepath = UPLOAD_FOLDER / temp_filename
            file.save(str(pred_filepath))
            predictor_filepaths.append(str(pred_filepath))
            temp_files.append(pred_filepath)
    # Selected files
    for pred_path in predictor_paths:
        if pred_path.strip():
            pred_filepath = str(PROJECT_ROOT / pred_path.replace('/', os.sep))
            if Path(pred_filepath).exists():
                predictor_filepaths.append(pred_filepath)
            else:
                return f"Error: Predictor file does not exist: {pred_path}", 400

    if not predictor_filepaths:
        return "Error: Must provide at least one predictor file", 400

    # Run prediction
    result_data = run_prediction_script(target_path_final, predictor_filepaths, model_choice, unique_id)

    # Clean up temp files after run
    for p in temp_files:
        try:
            os.remove(str(p))
        except Exception as e:
            print(f"[WARNING] Could not delete temp file {p}: {e}")

    if result_data['status'] == 'SUCCESS':
        return redirect(url_for('results_page', run_id=unique_id))
    else:
        return render_template('error.html', data=result_data), 500


@app.route('/results/<run_id>')
def results_page(run_id):
    metrics_path = OUTPUT_FOLDER / run_id / 'metrics.csv'
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics_raw = metrics_df.iloc[0].to_dict()
        # Ensure numeric fields are actually numeric, not strings
        metrics = {}
        numeric_fields = ['r2', 'pearson_r', 'spearman_r', 'rmse', 'n_samples', 'n_pixels']
        for key, value in metrics_raw.items():
            if key in numeric_fields:
                try:
                    metrics[key] = float(value) if pd.notna(value) else None
                except (ValueError, TypeError):
                    metrics[key] = None
            else:
                metrics[key] = value
        is_prediction = 'best_model' in metrics
        best_model_name = metrics.get('best_model', '')
        is_tree_model = best_model_name in ['RF', 'GBM', 'XGB'] 
        scatter_filename = 'parity.png' if is_prediction else 'scatter.png'
        scatter_url = url_for('static', filename=f"results/{run_id}/{scatter_filename}") 
        residuals_url = url_for('static', filename=f"results/{run_id}/residuals.png") 
        importance_url = url_for('static', filename=f"results/{run_id}/feature_importance.png")
        return render_template('results.html', 
                               run_id=run_id, 
                               metrics=metrics, 
                               scatter_url=scatter_url, 
                               residuals_url=residuals_url,
                               importance_url=importance_url,
                               is_prediction=is_prediction,
                               is_tree_model=is_tree_model) 
    return render_template('error.html', data={'error': f"Results not found for run ID {run_id}."}), 404



if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"Flask App Starting")
    print(f"{'='*60}")
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Folder: {DATA_FOLDER}")
    print(f"Available .tif files: {len(get_available_tif_files())}")
    print(f"{'='*60}\n")
    
    app.run(debug=True)