# 이미지 찍고 전송, 서버에서 처리됐다 하면 지우는 역할
# app 실행 후 capture_imgs 실행하기
from flask import Flask, request, jsonify, send_from_directory, send_file
from process import download_images
import os

app = Flask(__name__)

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


def process_images(anime_num, source_img_path):
    reference_img_path = f'input_img_examples/anime_{anime_num}.jpg'  # 연구컴에서 경로 확인
    # anime_num이 1이면 anime_1.jpg 2면 anime_2.jpg.. 5까지
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
        return save_file_path

@app.route('/animate', methods=['POST', 'GET'])
def animate():
    if request.method == 'POST':
        f = request.files['image']
        filepath = 'downloaded_files/img1.jpg'
        f.save(filepath)
        anime_num = int(request.form['anime_num']) # 나중에 form 안에 animenum 전송
        processed_image_path = process_images(anime_num, filepath) # 'result/img1_anigan.jpg'
        # 이미 torch save_img 로 저장했고,
        return send_file(processed_image_path, mimetype='image/jpg')
    elif request.method == 'GET':
        return send_file('result/img1_anigan.jpg', mimetype='image/jpg')


if __name__ == '__main__':
    app.run()
    # host='192.168.1.19', port=5000