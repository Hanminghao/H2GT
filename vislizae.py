import pickle
import openslide
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

pkl_path = '/home/hmh/project/my_secpro/plots/HoverNet_cTrans/BRCA/my_kfold_10/1/vislization.pkl'
svs_path = '/home/dataset2/hmh_data/TCGA_raw/BRCA/'
zoom = 16
with open(pkl_path,"rb") as file:
    data = pickle.load(file)
for case_name in tqdm(data.keys()):
    slide = openslide.open_slide(os.path.join(svs_path, case_name[0]+'.svs'))
    best_level = slide.get_best_level_for_downsample(zoom+1)
    if abs(slide.level_downsamples[best_level] - zoom) > 1:
        print(case_name[0],'error')
        continue

    width = slide.level_dimensions[best_level][0]
    height = slide.level_dimensions[best_level][1]
    MAG = int(float(slide.properties['aperio.AppMag']))
    if MAG == 40:
        patch_step = 1024 // zoom
    elif MAG == 20:
        patch_step = 512 // zoom

    image = slide.read_region((0,0), level=best_level, size=(width, height))
    new_data = data[case_name]

    # 提取坐标和评分
    patch_coordinates = np.array([coord for coord, _, _, _ in new_data])
    patch_scores = np.array([score for _, score, _, _ in new_data])
    event_time = np.arry(new_data[2])
    risk = np.arry(new_data[3])

    # 创建与切片图像相同大小的空白热图
    heatmap = np.zeros((height, width))

    # 将分数数据映射到热图上
    for position, score in zip(patch_coordinates, patch_scores):
        y, x = int(position[0]//zoom), int(position[1]/zoom)
        heatmap[x:x+patch_step,y:y+patch_step] = score
    # 叠加热图到切片图像
    alpha = 1  # 热图透明度
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # 标准化热图
    # 自定义颜色映射
    cmap = plt.cm.get_cmap('RdYlBu')  # 使用红-黄-蓝颜色映射

    # 将标准化后的热图应用到颜色映射
    heatmap_colored = cmap(heatmap)
    # 将透明度应用到热图
    heatmap_colored[:, :, 3] = alpha
    heatmap_colored[heatmap == 0] = [0, 0, 0, 0]

    overlay_image = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
    result = Image.alpha_composite(image.convert("RGBA"), overlay_image) 
    result.save(os.path.join('/home/dataset2/hmh_data/my_secpro_data/vislization/BRCA/', case_name[0]+'.png'))
    # overlay = np.dstack((heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap), np.full_like(heatmap, alpha)))
    # overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))
    # result = Image.blend(image.convert("RGBA"), overlay_image, alpha=alpha)
    # result.save(os.path.join('/home/hmh/', case_name[0]+'.png'))
    
print(1)
