#!/usr/bin/env python
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', default='runs/train/exp/weights/best.pt', help='模型路径')
    parser.add_argument('-s', '--source', default='datasets/flower/images/val', help='图片/文件夹/摄像头号')
    parser.add_argument('--img', type=int, default=640, help='推理尺寸')
    parser.add_argument('--conf', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--save', action='store_true', help='是否保存结果')
    return parser.parse_args()

def main(opt):
    model = YOLO(opt.weights)
    results = model.predict(source=opt.source,
                            imgsz=opt.img,
                            conf=opt.conf,
                            save=opt.save,
                            line_thickness=2,
                            show=True)          # 弹出窗口实时看
    print('✅ 推理完成，结果保存在 runs/detect/')

if __name__ == '__main__':
    main(parse_opt())