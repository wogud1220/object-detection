import json
import os

def change_bbox(json_path, update_dict):
    """
    ann_id별로 bbox 좌표를 수정하고 area를 자동 갱신하는 함수
    """
    print("🔧 change_box 실행 중...")

    if not os.path.exists(json_path):
        print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    hit = 0
    for ann in coco["annotations"]:
        ann_id = ann["id"]
        if ann_id in update_dict:
            old_bbox = ann["bbox"]
            new_bbox = update_dict[ann_id]
            ann["bbox"] = new_bbox
            ann["area"] = int(new_bbox[2] * new_bbox[3])
            hit += 1
            print(f"✅ ann_id={ann_id}: {old_bbox} → {new_bbox}, area={ann['area']}")

    if hit == 0:
        print("⚠️ update_dict에 해당하는 ann_id를 찾지 못했습니다.")
    else:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        print(f"\n💾 수정 완료 및 저장: {json_path}")
        print(f"🔁 총 {hit}개의 bbox가 업데이트되었습니다.")