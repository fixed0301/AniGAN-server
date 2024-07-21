# 연구컴 코드
# download_files 지우는 과정도 추가
import cv2
import requests
import os
import torch
from PIL import Image

#from torchvision import transforms
#from torchvision.utils import save_image
#from src.trainer import Trainer
#from src.utils import get_config

server_url = 'http://172.16.82.127:5000'
download_folder = 'downloaded_files'
if not os.path.exists(download_folder): # 왠지 모르겠지만 폴더를 생성하면서 사진을 다운받는다
    os.makedirs(download_folder)

result_folder = 'image2send'
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

frame_path = 'frame_0.png' # 나중에 인풋받음
reference_img_path = 'input_img_examples/1795111.png' # 연구컴에서 경로 확인


def cropface(img_path):
    faces = RetinaFace.detect_faces(img_path)
    img = Image.open(img_path).convert("RGB")

    for faceNum in faces.keys():
        identity = faces[faceNum]
        facial_area = identity["facial_area"]

        # 얼굴 영역을 강조 (직사각형 그리기)
        draw = ImageDraw.Draw(img)
        draw.rectangle([facial_area[0], facial_area[1], facial_area[2], facial_area[3]], outline="white", width=2)

        # 얼굴 영역을 잘라내기
        facial_img = img.crop((facial_area[0], facial_area[1], facial_area[2], facial_area[3]))

        return facial_img

def _denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def process_images(image_paths, output_path):
    
    config_file = r'C:\Users\Administrator\Desktop\AniGAN\AniGAN-main\src\configs\try4_final_r1p2.yaml'
    config = get_config(config_file)
    trainer = Trainer(config)
    trainer.cuda()

    ckpt_path = r'C:\Users\Administrator\Desktop\AniGAN\AniGAN-main\src\checkpoints\pretrained_face2anime.pt'
    trainer.load_ckpt(ckpt_path)
    trainer.eval()

    transform_list = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)


    source_img_path = image_paths[0]
    source_img = Image.open(source_img_path).convert('RGB')
    cropped_source_img = cropface(source_img)
    reference_img = Image.open(reference_img_path).convert('RGB')
    content_tensor = transform(cropped_source_img).unsqueeze(0).cuda()
    reference_tensor = transform(reference_img).unsqueeze(0).cuda()

    with torch.no_grad():
        generated_img = trainer.model.evaluate_reference(content_tensor, reference_tensor)
        name_part, ext_part = os.path.splitext(os.path.basename(source_img_path))
        save_file_name = f"{name_part}_anigan{ext_part}"
        save_file_path = os.path.join(output_dir, save_file_name)
        save_image(_denorm(generated_img), save_file_path, nrow=1, padding=0)
        print(f"Result is saved to: {save_file_path}")

        with open(save_file_path, 'rb') as img_file:
            response = requests.post(f'{server_url}/upload', files={'file': img_file})
            if response.status_code == 200:
                print(f"Successfully uploaded {save_file_path}")
            else:
                print(f"Failed to upload {save_file_path}")

def download_images():
    response = requests.get(f'{server_url}/files')
    if response.status_code == 200:
        files = response.json().get('files', [])
        if len(files) > 1:
            print("Too many files to animate")

        files = files[:]  # 근데 파일이 서버 업로드되고 바로 지워지는지 확인

        image_paths = []
        downloaded_files = []
        for file_name in files:
            file_response = requests.get(f'{server_url}/files/{file_name}')
            if file_response.status_code == 200:
                file_path = os.path.join(download_folder, file_name)
                with open(file_path, 'wb') as file:
                    file.write(file_response.content)
                image_paths.append(file_path)
                downloaded_files.append(file_name)  # 다운로드된 파일 이름 추가
                print(f"Downloaded {file_name} to {file_path}")
            else:
                print(f"Failed to download {file_name}")

        #process_images(image_paths, result_folder)
    else:
        print("Failed to retrieve file list from server")


if __name__ == '__main__':
    app.run()