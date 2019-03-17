import numpy as np


def nms(boxes, threshold):
        # 所有box的左上角、右下角坐标、以及每个box的score
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]

    # 每一个候选框的面积,之所+1是因为一个像素点的面积是1
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将score降序排列，得到的是索引值
    order = score.argsort()[::-1]

    keeps = []
    while order.size > 0:
        max_idx = order[0]
        keeps.append(max_idx)

        # 计算score最大的框与所有剩余框相交部分的坐标
        xx1 = np.maximum(x1[max_idx], x1[order[1:]])
        yy1 = np.maximum(y1[max_idx], y1[order[1:]])
        xx2 = np.minimux(x2[max_idx], x2[order[1:]])
        yy2 = np.minimux(y2[max_idx], y2[order[1:]])

        # 计算相交部分的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算IoU，inter / area1 + area2 - inter
        iou = inter / (areas[max_idx] + areas[order[1:]] - inter)

        # 删除IoU大于阈值的box，保留小于阈值的box，用这些box更新order
        # 如果order长度为n, 则iou长度是n-1(n-1个box和第一个box的iou), 求出小于阈值的box在iou中的位置再+1
        # 就是这个box在oder中的位置
        idx = np.where(iou <= threshold)[0]
        order = order[idx + 1]

    return keeps
