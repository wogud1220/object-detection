import os
from pathlib import Path

"""
    root_dir 하위의 이미지와 json을 읽어들여서, 파일 갯수를 구한다.
    그런데, 필요 없을듯.....
"""

json_cnt = 0

def check_data(root_dir="C:/workspace/github/data"):
    # 구글 코랩과 이미지 파일/디렉토리가 있는 루트 디렉토리 설정
    #root_dir = "C:/workspace/github/data/"

    # 접근할 파일/디렉토리 설정

    # 디렉토리 함수
    train_dir      = Path(root_dir + '/train_images')
    test_dir       = Path(root_dir + '/test_images')
    annotation_dir = Path(root_dir + '/train_annotations')

    # 파일 읽기
    train_images = sorted(train_dir.glob('*.png'))
    test_images  = sorted(test_dir.glob('*.png'))
    #annotation_normal_json = sorted(annotation_dir.glob('*.json'))

    # 출력
    print('=== 데이터 확인 ===')
    print('Number of Images in train_images  :', len(train_images))
    print('Number of Images in test_images   :', len(test_images))
    #print('Number of Json in annotation_normal_json :', len(annotation_normal_json))

    global json_cnt
    json_cnt = 0
    json_count = search_json(annotation_dir)
    print('Number of Json in annotation, json_count :', json_count)

"""
json은 하위의 하위 디렉토리까지 읽어들여야 해서, 재귀적 호출을 사용한다.
# find ./ -type f -name '*.json' | wc -l
"""
def search_json(dirname):
    global json_cnt
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search_json(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.json':
                    json_cnt += 1
                    #print(full_filename)
    except PermissionError:
        pass

    #print('json_cnt : ', json_cnt)

    return json_cnt

def main():
    check_data()

if __name__ == "__main__":
    main()