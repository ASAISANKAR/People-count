from flask import Flask, request, render_template, send_from_directory
import os

app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__))

@app.route('/')
def index():
    return render_template('index.html')


import subprocess
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file part', 400
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    save_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
    file.save(save_path)

    # Check if file saved
    if not os.path.exists(save_path):
        return 'File not saved', 500

    try:
        result = subprocess.run(
            ['python3', 'test.py'],
            cwd=UPLOAD_FOLDER,
            capture_output=True,
            text=True,
            timeout=300  
        )
        if result.returncode != 0:
            return f'Error running test.py:<br><pre>{result.stderr}</pre>', 500
        return 'File uploaded and processed. You can now view the processed video below.'
    except subprocess.TimeoutExpired:
        return 'Processing timed out.', 500
    except Exception as e:
        return f'Unexpected error: {str(e)}', 500



@app.route('/input.mp4')
def serve_video():
    return send_from_directory(UPLOAD_FOLDER, 'input.mp4')


@app.route('/output.mp4')
def serve_output_video():
    return send_from_directory(UPLOAD_FOLDER, 'output.mp4', mimetype='video/mp4')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)