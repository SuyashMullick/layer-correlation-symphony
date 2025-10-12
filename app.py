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

    print(f"\n[DEBUG] Running command:\n  {' '.join(command)}\n")

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
        stats = pd.read_csv(metrics_path).iloc[0].to_dict() if metrics_path.exists() else {'error': 'Metrics file not generated.'}

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


def run_prediction_script(target_file: str, predictor_files: list, model: str, output_id: str) -> dict:
    python_check = _check_python()
    if python_check: return python_check

    run_output_dir = OUTPUT_FOLDER / output_id
    run_output_dir.mkdir(exist_ok=True)

    abs_target = str(Path(target_file).resolve())
    abs_predictors = [str(Path(p).resolve()) for p in predictor_files]
    abs_output = str(run_output_dir.resolve())

    command = [
        str(PYTHON_EXEC), '-m', 'cli.symph-predict',
        '--target', abs_target,
        '--predictors', *abs_predictors,
        '--out', abs_output,
        '--model', model,
        '--transform_y', 'log1p',
        '--sample', '200000'
    ]

    print(f"\n[DEBUG] Running prediction command:\n  {' '.join(command)}\n")

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
        stats = pd.read_csv(metrics_path).iloc[0].to_dict() if metrics_path.exists() else {'error': 'Metrics file not generated.'}

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
    return render_template('predict.html', model_choices=model_choices)


@app.route('/run_compare', methods=['POST'])
def run_compare():
    file_path_1 = request.form.get('file_path_1', '').strip()
    file_path_2 = request.form.get('file_path_2', '').strip()
    uploaded_file1 = request.files.get('user_file1')
    uploaded_file2 = request.files.get('user_file2')
    
    unique_id = str(int(time.time()))
    temp_files = []
    
    try:
        def get_filepath(uploaded_file, file_path, prefix):
            if uploaded_file and uploaded_file.filename:
                temp_filename = f"{unique_id}_{prefix}_{secure_filename(uploaded_file.filename)}"
                filepath = UPLOAD_FOLDER / temp_filename
                uploaded_file.save(str(filepath))
                temp_files.append(filepath)
                return str(filepath)
            elif file_path:
                filepath = PROJECT_ROOT / file_path.replace('/', os.sep)
                if not filepath.exists():
                    raise FileNotFoundError(f"Selected file does not exist: {file_path}")
                return str(filepath)
            return None

        filepath1 = get_filepath(uploaded_file1, file_path_1, 'A')
        filepath2 = get_filepath(uploaded_file2, file_path_2, 'B')
        
        if not filepath1 or not filepath2:
            return "Error: Must provide both files (either select or upload)", 400
        
        result_data = run_comparison_script(filepath1, filepath2, unique_id)
        
        if result_data['status'] == 'SUCCESS':
            return redirect(url_for('results_page', run_id=unique_id))
        else:
            return render_template('error.html', data=result_data), 500

    except FileNotFoundError as e:
        return str(e), 400
    finally:
        for temp_file in temp_files:
            try: os.remove(str(temp_file))
            except Exception as e: print(f"[WARNING] Could not delete temp file {temp_file}: {e}")


@app.route('/run_predict', methods=['POST'])
def run_predict():
    target_file = request.files.get('target_file')
    predictor_files = request.files.getlist('predictor_files')
    model_choice = request.form.get('model_choice')
    unique_id = str(int(time.time()))
    temp_files = []

    try:
        if not (target_file and target_file.filename):
            return "Error: Must provide a target file", 400

        temp_filename = f"{unique_id}_T_{secure_filename(target_file.filename)}"
        target_filepath = UPLOAD_FOLDER / temp_filename
        target_file.save(str(target_filepath))
        temp_files.append(target_filepath)
        target_path_final = str(target_filepath)

        predictor_filepaths = []
        for i, file in enumerate(predictor_files):
            if file and file.filename:
                temp_filename = f"{unique_id}_P{i}_{secure_filename(file.filename)}"
                pred_filepath = UPLOAD_FOLDER / temp_filename
                file.save(str(pred_filepath))
                predictor_filepaths.append(str(pred_filepath))
                temp_files.append(pred_filepath)

        if not predictor_filepaths:
            return "Error: Must provide at least one predictor file", 400

        result_data = run_prediction_script(target_path_final, predictor_filepaths, model_choice, unique_id)

        if result_data['status'] == 'SUCCESS':
            return redirect(url_for('results_page', run_id=unique_id))
        else:
            return render_template('error.html', data=result_data), 500
            
    finally:
        for p in temp_files:
            try: os.remove(str(p))
            except Exception as e: print(f"[WARNING] Could not delete temp file {p}: {e}")


@app.route('/results/<run_id>')
def results_page(run_id):
    metrics_path = OUTPUT_FOLDER / run_id / 'metrics.csv'
    
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.iloc[0].to_dict()
        
        is_prediction = 'best_model' in metrics
        best_model_name = metrics.get('best_model', '')
        is_tree_model = best_model_name in ['RF', 'GBM', 'XGB'] 
        
        plotly_parity_url = url_for('static', filename=f"results/{run_id}/parity.html") 
        plotly_residuals_url = url_for('static', filename=f"results/{run_id}/residuals.html") 
        plotly_scatter_url = url_for('static', filename=f"results/{run_id}/scatter.html") 
        importance_url = url_for('static', filename=f"results/{run_id}/feature_importance.png")
        heatmap_url = url_for('static', filename=f"results/{run_id}/predictor_heatmap.png")
                
        return render_template('results.html', 
                               run_id=run_id, 
                               metrics=metrics, 
                               plotly_parity_url=plotly_parity_url,
                               plotly_residuals_url=plotly_residuals_url,
                               plotly_scatter_url=plotly_scatter_url,
                               importance_url=importance_url,
                               heatmap_url=heatmap_url,
                               is_prediction=is_prediction,
                               is_tree_model=is_tree_model) 
    
    return render_template('error.html', data={'error': f"Results not found for run ID {run_id}."}), 404

if __name__ == '__main__':
    print(f"\n{'='*60}\nFlask App Starting\n{'='*60}")
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Folder: {DATA_FOLDER}")
    print(f"Available .tif files: {len(get_available_tif_files())}\n{'='*60}\n")
    app.run(debug=True)
