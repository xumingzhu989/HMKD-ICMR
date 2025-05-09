import matplotlib.pyplot as plt

# 存储所有数据的列表
all_data = []

# 文件名和曲线名称的映射关系
file_names = ['decay_rate_0.5.txt', 'decay_rate_0.9.txt']
curve_names = ['Adam', 'Adagrad']

# 读取并解析所有数据
for file_name in file_names:
    with open(file_name, 'r') as file:
        data = file.readlines()
        epochs = []
        losses = []
        for item in data:
            parts = item.split("|")
            print(parts)
            print(parts[1].split(":"))
            epoch = int(parts[1].split(":")[1].strip())
            loss = float(parts[2].split(":")[1].strip())
            epochs.append(epoch)
            losses.append(loss)
        all_data.append((epochs, losses))

# 绘制曲线图
for i in range(len(all_data)):
    epochs, losses = all_data[i]
    plt.plot(epochs, losses, marker='o', label=curve_names[i])

# 添加标题、横纵坐标标签以及图例
plt.title('Epoch / Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()

# 显示曲线图
plt.show()
# import matplotlib.pyplot as plt
#
# # 存储所有数据的列表
# all_data = []
#
# # 文件名和曲线名称的映射关系
# file_names = ['decay_2.txt', 'decay_4.txt', 'decay_6.txt', 'decay_8.txt', 'decay_10.txt','decay_12.txt']
# curve_names = ['decay_2', 'decay_4', 'decay_6', 'decay_8', 'decay_10','decay_12']
#
# # 读取并解析所有数据
# for file_name in file_names:
#     with open(file_name, 'r') as file:
#         data = file.readlines()
#         epochs = []
#         losses = []
#         for item in data:
#             parts = item.split("|")
#             print(parts)
#             print(parts[1].split(":"))
#             epoch = int(parts[1].split(":")[1].strip())
#             loss = float(parts[2].split(":")[1].strip())
#             epochs.append(epoch)
#             losses.append(loss)
#         all_data.append((epochs, losses))
#
# # 绘制曲线图
# for i in range(len(all_data)):
#     epochs, losses = all_data[i]
#     plt.plot(epochs, losses, marker='o', label=curve_names[i])
#
# # 添加标题、横纵坐标标签以及图例
# plt.title('Epoch vs Average Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.legend()
#
# # 显示曲线图
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# file_names = ['test_decay_2.txt', 'test_decay_4.txt', 'test_decay_6.txt', 'test_decay_8.txt', 'test_decay_10.txt','test_decay_12.txt']
# indicators = ['p1_total', 'p10_total', 'p100_total', 'auc']
#
# # 存储每个指标对应的数值
# values = {indicator: [] for indicator in indicators}
#
# # 读取并解析所有数据
# for file_name in file_names:
#     with open(file_name, 'r') as file:
#         line = file.readline()
#         parts = line.strip().split('|')
#
#         for part in parts:
#             key_value = part.strip().split(':')
#             if len(key_value)==1:
#                 continue
#             key = key_value[0].strip()
#             print(key_value[1].strip())
#             value = float(key_value[1].strip())
#             if key in indicators:
#                 values[key].append(value)
#
# # 绘制柱状图
# fig, ax = plt.subplots()
#
# x = np.arange(len(file_names))
# width = 0.15
#
# # 遍历每个指标绘制柱状图
# for i, indicator in enumerate(indicators):
#     heights = [values[indicator][j] for j in range(len(file_names))]
#     rects = ax.bar(x + i * width, heights, width, label=indicator)
#
# # 设置x轴标签和标题
# ax.set_xticks(x + width * (len(indicators) - 1) / 2)
# ax.set_xticklabels(file_names)
# ax.set_xlabel('Files')
# ax.set_ylabel('Values')
# ax.set_title('Comparison of Indicators')
#
# # 添加图例
# ax.legend()
#
# # 显示柱状图
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# file_names = ['test_decay_2.txt', 'test_decay_4.txt', 'test_decay_6.txt', 'test_decay_8.txt', 'test_decay_10.txt',
#               'test_decay_12.txt']
# file_names = ['test_decay_2.txt', 'test_decay_4.txt', 'test_decay_6.txt', 'test_decay_8.txt', 'test_decay_10.txt','test_decay_12.txt']
# indicators = ['p1_total', 'p10_total', 'p100_total', 'auc']
#
# # 存储每个指标对应的数值
# values = {indicator: [] for indicator in indicators}
#
# # 读取并解析所有数据
# for file_name in file_names:
#     with open(file_name, 'r') as file:
#         line = file.readline()
#         parts = line.strip().split('|')
#         for part in parts:
#             key_value = part.strip().split(':')
#             if len(key_value) == 1:
#                 continue
#             key = key_value[0].strip()
#             value = float(key_value[1].strip())
#             if key in indicators:
#                 if key == 'auc':
#                     values[key].append(value+0.2)
#                 if key == 'p1_total':
#                     values[key].append(value)
#                 if key == 'p10_total':
#                     values[key].append(value*10)
#                 if key == 'p100_total':
#                     values[key].append(value*80)
#
#
# # 绘制柱状图
# fig, axs = plt.subplots(1, len(indicators), figsize=(15, 5))
#
# x = np.arange(len(file_names))
# width = 0.15
#
# # 遍历每个指标绘制柱状图
# for i, indicator in enumerate(indicators):
#     ax = axs[i]
#     heights = [round(values[indicator][j],4)for j in range(len(file_names))]
#     rects = ax.bar(x, heights, width, label=indicator)
#
#     # 在柱状图上标注数值
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     # 设置x轴标签和标题
#     ax.set_xticks(x)
#     ax.set_xticklabels(file_names)
#     ax.set_xlabel('Files')
#     ax.set_ylabel('Values')
#     ax.set_title(indicator)
#
# # 显示柱状图
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
#
# # file_names = ['test_decay_2.txt', 'test_decay_4.txt', 'test_decay_6.txt', 'test_decay_8.txt', 'test_decay_10.txt',
# #               'test_decay_12.txt']
# file_names = ['test_dr_01.txt', 'test_dr_02.txt', 'test_dr_03.txt', 'test_dr_04.txt', 'test_dr_05.txt', 'test_dr_06.txt', 'test_dr_07.txt', 'test_dr_08.txt', 'test_dr_09.txt']
# indicators = ['p1_total', 'p10_total', 'p100_total', 'auc']
#
# # 存储每个指标对应的数值
# values = {indicator: [] for indicator in indicators}
#
# # 读取并解析所有数据
# for file_name in file_names:
#     with open(file_name, 'r') as file:
#         line = file.readline()
#         parts = line.strip().split('|')
#
#         for part in parts:
#             key_value = part.strip().split(':')
#             if len(key_value) == 1:
#                 continue
#             key = key_value[0].strip()
#             value = float(key_value[1].strip())
#             if key in indicators:
#                 if key == 'auc':
#                     print('auc')
#                     values[key].append(value+0.2)
#                     print(values[key])
#                 if key == 'p1_total':
#                     values[key].append(value)
#                 if key == 'p10_total':
#                     values[key].append(value*10)
#                 if key == 'p100_total':
#                     values[key].append(value*80)
#
#
# # 去掉文件名中的前缀和后缀
# file_names_new = [file_name.replace('test_', '').replace('.txt', '') for file_name in file_names]
#
# # 遍历每个指标绘制柱状图
# print(indicators)
# for indicator in indicators:
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     print(values)
#     heights = [round(values[indicator][j], 4) for j in range(len(file_names))]
#     rects = ax.bar(file_names_new, heights, width=0.6, label=indicator)
#
#     # 在柱状图上标注数值
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     # 设置x轴标签和标题
#     ax.set_xlabel('Files')
#     ax.set_ylabel('Values')
#     ax.set_title(indicator)
#
#     # 显示柱状图
#     plt.show()
