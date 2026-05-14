import cv2
import numpy as np
import os

def augment_image(image):
    augmented = []
    
    # 1. 좌우 반전
    flipped = cv2.flip(image, 1)
    augmented.append(('flip', flipped))
    
    # 2. 밝기 증가
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    augmented.append(('bright', bright))
    
    # 3. 밝기 감소
    dark = cv2.convertScaleAbs(image, alpha=0.7, beta=-30)
    augmented.append(('dark', dark))

    return augmented

def augment_label(label_path, aug_type):
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        
        if aug_type == 'flip':
            x = 1.0 - x
        # bright/dark는 라벨 그대로
        
        new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    return new_lines

# 경로 설정
base_path = r'C:\Users\samsungacademy608-1\Desktop\Dataset (2)\Dataset\object'
splits = ['train', 'val']

for split in splits:
    img_dir = os.path.join(base_path, 'images', split)
    lbl_dir = os.path.join(base_path, 'labels', split)
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n✅ {split}: {len(img_files)}개 → 증강 시작")
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        lbl_file = os.path.splitext(img_file)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_file)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        augmented = augment_image(image)
        
        for aug_type, aug_img in augmented:
            new_name = os.path.splitext(img_file)[0] + f'_{aug_type}'
            
            # 이미지 저장
            new_img_path = os.path.join(img_dir, new_name + '.jpg')
            cv2.imwrite(new_img_path, aug_img)
            
            # 라벨 저장
            new_lbl_path = os.path.join(lbl_dir, new_name + '.txt')
            new_lines = augment_label(lbl_path, aug_type)
            if new_lines:
                with open(new_lbl_path, 'w') as f:
                    f.writelines(new_lines)

print("\n🎉 데이터 증강 완료!")
print("기존 대비 4배 데이터로 늘어났어!")