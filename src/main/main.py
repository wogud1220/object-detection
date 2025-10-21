import os
import json
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import warnings

from collections import defaultdict

from ultralytics import YOLO
import yaml

from IPython.display import Image as IPImage, display

import pandas as pd
from collections import Counter

import torch

from src.data.PillDataset import PillDataset
from src.utils import util
from src.utils.albumentations_A import train_compose
from src.utils.albumentations_A import val_compose

# 데이터 기본 경로 (압축 해제한 위치)
root_dir="C:/workspace/github/data" #임시 경로. 경로 정해지면 삭제 필요함.
BASE_DIR = root_dir  #"/content/data" #이미지 경로는 여기에 설정.

# 학습 및 테스트 데이터 경로
TRAIN_IMG_DIR = f"{BASE_DIR}/train_images"
TRAIN_ANN_DIR = f"{BASE_DIR}/train_annotations"
TEST_IMG_DIR = f"{BASE_DIR}/test_images"

YOLO_DIR = f"{BASE_DIR}/yolo_dataset"

def main():

    """ main """

    """ # 경로 확인 """
    check_datapath()

    """ # 테스트 이미지 출력 """
    show_testimages(TEST_IMG_DIR)

    """ # Annotation 파일 수집 및 통합 """
    train_data, all_json_files = process_annotation(TRAIN_ANN_DIR)

    """ # 데이터 탐색 """
    images_df, categories_df, annotations_df = search_data(train_data)

    """ # 어노테이션 시각화 """
    process_visualize_annotations(images_df, categories_df, annotations_df)

    """ JSON 파일을 확인 """
    check_json(all_json_files)

    """ 데이터셋, 데이터로더 처리 """
    train_images_df, val_images_df, train_annotations_df, val_annotations_df = process_data(images_df, categories_df, annotations_df)

    """ YOLO 데이터셋 """
    category_id_mapping, num_classes = process_yolo_dataset(categories_df)

    """ # Train, Val 데이터 변환 """
    train_success, val_success = convert_data(train_images_df, val_images_df, train_annotations_df, val_annotations_df, category_id_mapping)

    """ 클래스 이름 리스트 생성 """
    yaml_path = make_class_list(categories_df, num_classes, train_success, val_success)

    """ 모델 인스턴스 생성 """
    model = make_model()

    """ 모델 학습 """
    make_train(model, yaml_path)

    """ 모델 결과 """
    predict_model()

def check_datapath():
    # 경로 확인
    print("📂 경로 설정 완료:")
    print(f"BASE_DIR      : {BASE_DIR}")
    print(f"TRAIN_IMG_DIR : {TRAIN_IMG_DIR}")
    print(f"TEST_IMG_DIR  : {TEST_IMG_DIR}")

    # 실제 폴더 및 파일 존재 여부 확인
    print("📂 경로 설정:")
    for name, path in [("BASE_DIR", BASE_DIR),
                       ("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
                       ("TRAIN_ANN_DIR", TRAIN_ANN_DIR),
                       ("TEST_IMG_DIR", TEST_IMG_DIR)]:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name}: {path}")

def show_testimages(TEST_IMG_DIR="/content/data/test_images"):
    # 테스트 이미지 폴더
    #TEST_IMG_DIR = "/content/data/test_images"

    # 파일 목록 불러오기
    test_files = sorted(os.listdir(TEST_IMG_DIR))

    # 이미지가 있는지 확인
    print(f"총 테스트 이미지 개수: {len(test_files)}")
    print("예시 파일명:", test_files[:5])

    # 앞부분 9개만 미리보기
    sample_files = test_files[:9]

    # 시각화
    plt.figure(figsize=(12, 12))
    for i, img_name in enumerate(sample_files):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(img_name, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Annotation 파일 수집 및 통합
def process_annotation(TRAIN_ANN_DIR="/content/data/train_annotations"):
    #TRAIN_ANN_DIR = "/content/data/train_annotations"

    # 모든 JSON 파일 찾기
    all_json_files = []
    for root, dirs, files in os.walk(TRAIN_ANN_DIR):
        for file in files:
            if file.endswith('.json'):
                all_json_files.append(os.path.join(root, file))

    print(f"✅ 총 JSON 파일 개수: {len(all_json_files)}")
    print(f"예시 파일:\n{all_json_files[0]}")

    # file_name을 키로 하여 데이터 수집
    images_dict = {}  # {file_name: image_info}
    annotations_by_image = defaultdict(list)  # {file_name: [annotations]}
    categories_dict = {}  # {category_id: category_name}

    print("\n📊 JSON 파일 처리 중...")
    for idx, json_path in enumerate(all_json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보 수집
            if 'images' in data and len(data['images']) > 0:
                img = data['images'][0]
                file_name = img['file_name']

                # 이미지 정보는 한 번만 저장 (중복 방지)
                if file_name not in images_dict:
                    images_dict[file_name] = {
                        'file_name': file_name,
                        'width': img.get('width'),
                        'height': img.get('height'),
                    }

            # Annotation 수집 (같은 file_name끼리 묶음)
            if 'annotations' in data:
                for ann in data['annotations']:
                    annotations_by_image[file_name].append({
                        'category_id': ann['category_id'],
                        'bbox': ann['bbox'],
                        'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                        'iscrowd': ann.get('iscrowd', 0)
                    })

            # 카테고리 수집
            if 'categories' in data:
                for cat in data['categories']:
                    categories_dict[cat['id']] = cat['name']

            # 진행상황 출력 (500개마다)
            if (idx + 1) % 500 == 0:
                print(f"  처리 중... {idx + 1}/{len(all_json_files)}")

        except Exception as e:
            print(f"❌ 오류 ({os.path.basename(json_path)}): {e}")
            continue

    # COCO 형식으로 최종 정리
    combined_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    image_id = 0
    annotation_id = 0

    print("\n🔗 이미지와 Annotation 연결 중...")
    for file_name, img_info in images_dict.items():
        # 이미지 추가
        img_info['id'] = image_id
        combined_data['images'].append(img_info)

        # 해당 이미지의 모든 annotation 추가
        for ann in annotations_by_image[file_name]:
            combined_data['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann['iscrowd']
            })
            annotation_id += 1

        image_id += 1

    # 카테고리 정리
    combined_data['categories'] = [
        {'id': cat_id, 'name': cat_name}
        for cat_id, cat_name in sorted(categories_dict.items())
    ]

    print(f"\n✅ 통합 완료!")
    print(f"  - 총 이미지: {len(combined_data['images'])}")
    print(f"  - 총 Annotation: {len(combined_data['annotations'])}")
    print(f"  - 총 카테고리: {len(combined_data['categories'])}")
    print(f"  - 평균 이미지당 객체 수: {len(combined_data['annotations']) / len(combined_data['images']):.2f}개")

    # 통합 데이터 저장
    train_data = combined_data

    # 나중에 재사용할 수 있도록 파일로 저장
    output_path = f"{BASE_DIR}/train_combined.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    print(f"\n💾 통합 파일 저장: {output_path}")

    return train_data, all_json_files

def search_data(train_data):
    # 데이터 탐색

    # 이미지 정보
    images_df = pd.DataFrame(train_data['images'])
    print(f"📷 총 이미지 개수: {len(images_df)}")
    print(images_df.head())

    # 카테고리 정보
    categories_df = pd.DataFrame(train_data['categories'])
    print(f"\n🏷️ 총 카테고리(알약 종류): {len(categories_df)}")
    print(categories_df)

    # Annotation 정보
    annotations_df = pd.DataFrame(train_data['annotations'])
    print(f"\n📦 총 Annotation 개수: {len(annotations_df)}")
    print(annotations_df.head())

    # 카테고리별 분포
    category_counts = Counter(annotations_df['category_id'])
    print("\n📊 카테고리별 객체 개수:")
    for cat_id, count in sorted(category_counts.items()):
        cat_name = categories_df[categories_df['id'] == cat_id]['name'].values[0]
        print(f"  Class {cat_id} ({cat_name}): {count}개")

    # 이미지당 객체 수 분포
    img_obj_counts = annotations_df.groupby('image_id').size()
    print(f"\n📈 이미지당 객체 수 통계:")
    print(f"  - 평균: {img_obj_counts.mean():.2f}개")
    print(f"  - 최소: {img_obj_counts.min()}개")
    print(f"  - 최대: {img_obj_counts.max()}개")

    return images_df, categories_df, annotations_df

def setting_font():
    #path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'  # 나눔 고딕
    path = 'C:/Windows/Fonts/나눔고딕/NanumGothic.ttf'  # 나눔 고딕
    font_name = fm.FontProperties(fname=path, size=10).get_name()  # 기본 폰트 사이즈 : 10
    plt.rc('font', family=font_name)

    fm.fontManager.addfont(path)

def process_visualize_annotations(images_df, categories_df, annotations_df):
    valid_image_ids = annotations_df['image_id'].unique()
    print(f"📊 Annotation이 있는 이미지: {len(valid_image_ids)}개")
    print(f"📊 전체 이미지: {len(images_df)}개")

    if len(valid_image_ids) < len(images_df):
        print(f"⚠️ Annotation이 없는 이미지: {len(images_df) - len(valid_image_ids)}개")

    # 객체 수별 분포 다시 확인
    img_obj_counts_df = annotations_df.groupby('image_id').size().reset_index(name='count')
    print(f"\n📊 객체 수별 이미지 분포:")
    print(img_obj_counts_df['count'].value_counts().sort_index())

    # 여러 객체가 있는 이미지 찾기
    multi_obj_images = img_obj_counts_df[img_obj_counts_df['count'] >= 2]
    print(f"\n✅ 2개 이상 객체: {len(multi_obj_images)}개")

    # 시각화
    print("\n🎨 샘플 이미지 시각화 (크기 조정):")

    if len(multi_obj_images) > 0:
        # 여러 객체가 있는 이미지 우선
        print("여러 객체가 있는 이미지:")
        sample_ids = multi_obj_images['image_id'].sample(min(3, len(multi_obj_images))).values
    else:
        # 없으면 랜덤
        print("랜덤 샘플:")
        sample_ids = img_obj_counts_df['image_id'].sample(min(3, len(img_obj_counts_df))).values

    for img_id in sample_ids:
        util.visualize_annotations(TRAIN_IMG_DIR,
                          images_df,
                          annotations_df,
                          categories_df,
                                   img_id, figsize=(8, 8))

def check_json(all_json_files):
    #  원본 JSON 파일에서 직접 확인

    # Image ID 1023의 파일명으로 원본 JSON 찾기
    target_file = "K-001900-016548-031705-033208_0_2_0_2_75_000_200.png"

    print(f"🔍 {target_file}에 해당하는 원본 JSON 파일들:\n")

    json_count = 0
    for json_path in all_json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' in data and len(data['images']) > 0:
                if data['images'][0]['file_name'] == target_file:
                    json_count += 1
                    print(f"[{json_count}] {os.path.basename(json_path)}")

                    if 'annotations' in data:
                        for ann in data['annotations']:
                            cat_id = ann['category_id']
                            # categories에서 이름 찾기
                            cat_name = "Unknown"
                            if 'categories' in data:
                                for cat in data['categories']:
                                    if cat['id'] == cat_id:
                                        cat_name = cat['name']
                                        break
                            bbox = ann['bbox']
                            print(f"    - {cat_name} (ID: {cat_id})")
                            print(f"      BBox: {bbox}")
                    print()
        except:
            continue

    print(f"✅ 총 {json_count}개의 JSON 파일 발견")
    print(f"\n💡 결론: 원본 데이터에도 {json_count}개의 annotation만 있음")
    print("    → 병합 과정은 정상이며, 데이터셋 자체가 이렇게 제공")

def process_data(images_df, categories_df, annotations_df):
    # Train/Val Split
    train_ids, val_ids = train_test_split(
        images_df['id'].values,
        test_size=0.2,
        random_state=42
    )

    train_images_df = images_df[images_df['id'].isin(train_ids)].reset_index(drop=True)
    val_images_df = images_df[images_df['id'].isin(val_ids)].reset_index(drop=True)

    train_annotations_df = annotations_df[annotations_df['image_id'].isin(train_ids)]
    val_annotations_df = annotations_df[annotations_df['image_id'].isin(val_ids)]

    train_transform = train_compose()
    val_transform = val_compose()

    # Dataset 생성
    train_dataset = PillDataset(
        TRAIN_IMG_DIR,
        train_images_df,
        train_annotations_df,
        categories_df,
        transform=train_transform
    )

    val_dataset = PillDataset(
        TRAIN_IMG_DIR,
        val_images_df,
        val_annotations_df,
        categories_df,
        transform=val_transform
    )

    # # Collate 함수
    # def collate_fn(batch):
    #     return tuple(zip(*batch))

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    print("✅ 데이터 증강이 적용된 Dataset/DataLoader 생성 완료!")
    print(f"  - Train: {len(train_dataset)}개")
    print(f"  - Val: {len(val_dataset)}개")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")

    # 샘플 확인
    images, targets = next(iter(train_loader))
    print(f"\n✅ 샘플 배치:")
    print(f"  - Batch size: {len(images)}")
    print(f"  - 이미지 shape: {images[0].shape}")
    print(f"  - 객체 수: {len(targets[0]['labels'])}개")

    return train_images_df, val_images_df, train_annotations_df, val_annotations_df

# Collate 함수
def collate_fn(batch):
    return tuple(zip(*batch))

def process_yolo_dataset(categories_df):
    # YOLO 데이터셋 폴더 구조 생성
    #YOLO_DIR = f"{BASE_DIR}/yolo_dataset"
    os.makedirs(f"{YOLO_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/val", exist_ok=True)

    print("✅ YOLO 폴더 구조 생성 완료!")

    # 카테고리 ID를 0부터 시작하도록 매핑
    category_id_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted(categories_df['id'].unique()))}
    num_classes = len(category_id_mapping)

    print(f"📊 총 클래스 수: {num_classes}개")
    print(f"카테고리 매핑 (처음 5개): {dict(list(category_id_mapping.items())[:5])}")

    return category_id_mapping, num_classes

def convert_data(train_images_df, val_images_df, train_annotations_df, val_annotations_df, category_id_mapping):
    # Train 데이터 변환
    print("📝 Train 데이터 변환 중...")
    train_success = 0
    for _, img_info in tqdm(train_images_df.iterrows(), total=len(train_images_df)):
        if util.convert_to_yolo_format(
                img_info,
                train_annotations_df,
                f"{YOLO_DIR}/images/train",
                f"{YOLO_DIR}/labels/train",
                category_id_mapping,
                TRAIN_IMG_DIR
        ):
            train_success += 1

    print(f"✅ Train 데이터 변환 완료: {train_success}/{len(train_images_df)}개")

    # Val 데이터 변환
    print("\n📝 Val 데이터 변환 중...")
    val_success = 0
    for _, img_info in tqdm(val_images_df.iterrows(), total=len(val_images_df)):
        if util.convert_to_yolo_format(
                img_info,
                val_annotations_df,
                f"{YOLO_DIR}/images/val",
                f"{YOLO_DIR}/labels/val",
                category_id_mapping,
                TRAIN_IMG_DIR
        ):
            val_success += 1

    print(f"✅ Val 데이터 변환 완료: {val_success}/{len(val_images_df)}개")

    return train_success, val_success

def make_class_list(categories_df, num_classes, train_success, val_success):
    # 클래스 이름 리스트 생성
    class_names = []
    for cat_id in sorted(categories_df['id'].unique()):
        cat_name = categories_df[categories_df['id'] == cat_id]['name'].values[0]
        class_names.append(cat_name)

    # data.yaml 내용
    data_yaml = {
        'path': YOLO_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': num_classes,
        'names': class_names
    }

    # 저장
    yaml_path = f"{YOLO_DIR}/data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print("✅ data.yaml 생성 완료!")
    print(f"경로: {yaml_path}")
    print(f"\n📋 설정 내용:")
    print(f"  - Train 이미지: {train_success}개")
    print(f"  - Val 이미지: {val_success}개")
    print(f"  - 클래스 수: {num_classes}개")
    print(f"  - 클래스 예시: {class_names[:3]}")

    return yaml_path

def make_model():
    model = YOLO('yolov8m.pt')
    return model

def make_train(model, yaml_path):
    # 학습 파라미터
    results = model.train(
        data=yaml_path,
        epochs=1,  ##20,  # 최대 20 에폭  ##임시로 에폭을 1로 설정함.
        imgsz=800,  # 이미지 크기
        batch=8,  # 배치 크기
        patience=10,  # Early stopping patience (10 에폭 동안 개선 없으면 중단)
        save=True,  # 모델 저장
        device='cpu', ##0 if torch.cuda.is_available() else 'cpu',  # GPU 자동 선택 ##임시로 cpu로 설정함.
        project=f'{BASE_DIR}/yolo_runs',  # 결과 저장 폴더
        name='pill_detection',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,  # 초기 learning rate
        lrf=0.01,  # 최종 learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        label_smoothing=0.0,
        val=True,  # Validation 수행
        plots=True,  # 학습 그래프 자동 생성
        verbose=True
    )

    print("\n 학습 완료!")
    print(f" 결과 저장 위치: {BASE_DIR}/yolo_runs/pill_detection")

def predict_model():
    # 결과 디렉터리 설정
    result_dir = f"{BASE_DIR}/yolo_runs/pill_detection"

    print("📈 YOLOv8 학습 결과 요약")
    print("───────────────────────────────")

    # 1️⃣ Loss 그래프
    loss_img = f"{result_dir}/results.png"
    if os.path.exists(loss_img):
        print("\n1. 🔹 Loss 변화 그래프")
        display(IPImage(filename=loss_img))
    else:
        print("❌ Loss 그래프(results.png)를 찾을 수 없습니다.")

    # 2️⃣ Confusion Matrix
    cm_img = f"{result_dir}/confusion_matrix.png"
    if os.path.exists(cm_img):
        print("\n2. 🔹 Confusion Matrix")
        display(IPImage(filename=cm_img))
    else:
        print("❌ Confusion Matrix 이미지를 찾을 수 없습니다.")

    # 3️⃣ Validation 예측 결과 샘플
    pred_img = f"{result_dir}/val_batch0_pred.jpg"
    if os.path.exists(pred_img):
        print("\n3. 🔹 Validation 예측 결과 샘플")
        display(IPImage(filename=pred_img))
    else:
        print("❌ 예측 결과 이미지(val_batch0_pred.jpg)를 찾을 수 없습니다.")

    # 4️⃣ Best 모델 경로
    best_model = f"{result_dir}/weights/best.pt"
    print(f"\n✅ Best 모델 경로:\n{best_model if os.path.exists(best_model) else '❌ 파일이 없습니다.'}")

if __name__ == "__main__":
    main()