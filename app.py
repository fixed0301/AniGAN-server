# 이미지 찍고 전송, 서버에서 처리됐다 하면 지우는 역할
# app 실행 후 capture_imgs 실행하기
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'photos2send'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 파일 업로드
@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return 'File successfully uploaded', 200

# 나 이거 업로드햇서
@app.route('/files', methods=['GET'])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({'files': files})

# 파일 다운로드
@app.route('/files/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 삭제할 파일 목록을 json으로 수신
# 원래 sada4cut 에서는 photos를 덮어씌웠는데, 여기는 photos를 서버에 업로드한 경로로 설정하다보니 올리고 바로 지운다.
# 찍은걸 다시 띄울 필요는 없으니
@app.route('/delete', methods=['POST'])
def delete_file():
    file_names = request.json.get('files', [])
    for file_name in file_names:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        if os.path.exists(file_path): # upload 폴더에 있는거 삭제
            os.remove(file_path)
    return 'Files successfully deleted', 200 # 다 삭제했다고 응답


if __name__ == '__main__':
    app.run(host='172.16.82.127', port=5000)
