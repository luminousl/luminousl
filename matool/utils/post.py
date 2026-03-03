import os
from flask import *
from werkzeug.utils import secure_filename
from loguru import logger
import os,sys
import datetime
import subprocess
import mago

pwd=os.path.dirname(os.path.abspath(__file__))
upload_root = '/media/cmcserver/disk/shared/MATool/'
app = Flask(__name__)

# 处理上传文件的路由
@app.route('/upload', methods=['GET','POST'])
def upload_file():    
    return render_template('upload.html')

def get_upload_root(dir_name):
    now = datetime.datetime.now()
    if now.second == 0 or now.second == 1:
        now = now - datetime.timedelta(minutes=1)
    now_str = now.strftime('%Y%m%d-%H%M')
    
    upload_report = f"{os.path.basename(dir_name)}_{now_str}"

    upload_report_root = os.path.join(upload_root,upload_report)
    os.system(f"mkdir -p {upload_report_root}")
    return upload_report_root

@app.route('/file_upload', methods=['GET','POST'])
def file_upload():
    file = request.files['file']    
    upload_report_root = get_upload_root(os.path.dirname(file.filename))
    file.save(os.path.join(upload_report_root, os.path.basename(file.filename)))
    return 'OK'
    
    
@app.route('/py_script', methods=['GET','POST']) 
def py_script():
    file = request.files['file']
    upload_report_root = get_upload_root(os.path.dirname(file.filename))
    mago.main_parse(model=upload_report_root) 
    return 'OK'
    

# 下载文件的路由
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8069,debug=True,host='0.0.0.0')
