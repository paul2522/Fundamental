import os
import pickle
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

# 전처리 시 생성했던 디렉토리 구조 = 이미지 파일 모아놓은곳
dir_path = '/home/aiffel/Code/AiffelPractice/Practice6'
train_file_path = os.path.join(dir_path, 'train')
images_dir_path = os.path.join(dir_path, 'cifar-images')

def get_histogram(image):
    histogram = []
    
    for i in range(3):
        channel_histogram = cv.calcHist(images = [image], channels = [i], mask = None, histSize=[4], ranges=[0,256])
        histogram.append(channel_histogram)

    histogram = np.concatenate(histogram)
    histogram = cv.normalize(histogram, histogram)

    return histogram

def build_histogram_db():
    histogram_db = {}

    path = images_dir_path
    file_list = os.listdir(images_dir_path)

    # 파일 1개씩
    for file_name in tqdm(file_list):
        file_path = os.path.join(images_dir_path, file_name)
        image = cv.imread(file_path)
        histogram = get_histogram(image)
        histogram_db[file_name] = histogram

    return histogram_db

histogram_db = build_histogram_db()

def get_target_histogram():
    filename = input("이미지 파일명을 입력하세요: ")
    if filename not in histogram_db:
        print('유효하지 않은 이미지 파일명입니다.')
        return None
    return histogram_db[filename]

# OpenCV의 compareHist() 함수를 사용하여 입력 이미지와 검색 대상 이미지 하나하나의 히스토그램 간 유사도를 계산한다. 결과는 result라는 이름의 딕셔너리로, 키는 이미지 이름, 값은 유사도로 한다.
# 계산된 유사도를 기준으로 정렬하여 순서를 매긴다.
# 유사도 순서상으로 상위 5개 이미지만 골라서 result에 남긴다.

def search(histogram_db, target_histogram, top_k=5):
    results = {}

    # Calculate similarity distance by comparing histograms.
    for file_name, histogram in tqdm(histogram_db.items()):
        value = cv.compareHist(histogram,target_histogram,0)
        results[file_name] = value

    results = dict(sorted(results.items(), key = lambda x: x[1])[:top_k])

    return results

def show_result(result):
    f=plt.figure(figsize=(10,3))
    for idx, filename in enumerate(result.keys()):    
        file_path = os.path.join(images_dir_path, filename)
        im = f.add_subplot(1,len(result),idx+1)
        img = Image.open(file_path)
        im.imshow(img)

# print(histogram_db['adriatic_s_001807.png'])
# adriatic_s_001807.png

target_histogram  = get_target_histogram()
result = search(histogram_db, target_histogram)
show_result(result)
