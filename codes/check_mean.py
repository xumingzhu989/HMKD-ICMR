# # 读取文本文件
#
# with open('check.txt', 'r') as file:
#     lines = file.readlines()
#
# # 提取Validation pixAcc和mIoU的值
# pix_acc_values = []
# miou_values = []
# for line in lines:
#     if "Overall validation pixAcc" in line:
#             parts = line.split(',')
#             pix_acc = float(parts[2].split(':')[-1])
#             miou = float(parts[3].split(':')[-1])
#             pix_acc_values.append(pix_acc)
#             miou_values.append(miou)
#
# # 计算均值
# avg_pix_acc = sum(pix_acc_values) / len(pix_acc_values)
# avg_miou = sum(miou_values) / len(miou_values)
#
# print("Validation pixAcc 平均值:", avg_pix_acc)
# print("mIoU 平均值:", avg_miou)
import re
import matplotlib.pyplot as plt

# 存储所有文件中的pixAcc和mIoU数值以及最大值的索引和数值
all_pixAcc_values = []
all_mIoU_values = []
max_pixAcc_values = []
max_pixAcc_iters = []
max_mIoU_values = []
max_mIoU_iters = []

# 循环处理每个文件
#file_names = ['H_V_KL.txt', 'H_V_MSE.txt', 'MA_KL.txt', 'MA_MSE.txt', 'MA_MSE_C.txt', 'ATT.txt', 'WO_p.txt']  # 替换为实际文件名
file_names = ['EGLA4.txt']
for file_name in file_names:
    with open(file_name, 'r') as file:
        data = file.read()
    matches = re.findall(r'Overall validation pixAcc: (\d+\.\d+), mIoU: (\d+\.\d+)', data)
    pixAcc_values = [float(match[0]) for match in matches]
    mIoU_values = [float(match[1]) for match in matches]
    all_pixAcc_values.append(pixAcc_values)
    all_mIoU_values.append(mIoU_values)

    max_pixAcc = max(pixAcc_values)
    max_mIoU = max(mIoU_values)

    max_pixAcc_index = pixAcc_values.index(max_pixAcc)
    max_mIoU_index = mIoU_values.index(max_mIoU)

    max_pixAcc_values.append(max_pixAcc)
    max_pixAcc_iters.append(max_pixAcc_index)
    max_mIoU_values.append(max_mIoU)
    max_mIoU_iters.append(max_mIoU_index)

# 绘制pixAcc折线图
iters = range(max(len(pixAcc) for pixAcc in all_pixAcc_values))  # 使用最大长度作为迭代器

plt.figure(figsize=(10, 5))
for i in range(len(file_names)):
    plt.plot(iters, all_pixAcc_values[i], label=f'pixAcc - {file_names[i]}')

# 标出最大值
for i in range(len(file_names)):
    #plt.text(max_pixAcc_iters[i], max_pixAcc_values[i], f"Max: {max_pixAcc_values[i]}", ha='right')
    print(file_names[i],max_pixAcc_iters[i], max_pixAcc_values[i])
plt.xlabel('Iterations(800)')
plt.ylabel('Value')
plt.title('pixAcc over Iterations')
plt.legend()
plt.show()

# 绘制mIoU折线图
plt.figure(figsize=(10, 5))
for i in range(len(file_names)):
    plt.plot(iters, all_mIoU_values[i], label=f'mIoU - {file_names[i]}')

# 标出最大值
for i in range(len(file_names)):
    #plt.text(max_mIoU_iters[i], max_mIoU_values[i], f"Max: {max_mIoU_values[i]}", ha='right')
    print(file_names[i],max_mIoU_iters[i], max_mIoU_values[i])
plt.xlabel('Iterations(800)')
plt.ylabel('Value')
plt.title('mIoU over Iterations')
plt.legend()
plt.show()
