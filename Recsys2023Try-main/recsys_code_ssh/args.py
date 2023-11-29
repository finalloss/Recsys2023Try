# 预先求得的类别特征信息
# feature_sizes = [23, 139, 5, 638, 6, 5234, 1, 6, 7, 3, 24, 26, 331, 19, 5854, 12, 49, 924, 19, 57, 35, 
                #  26, 4, 4, 3, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# 若未对离散特征进行编码
# feature_sizes = [68, 32684, 22295, 32746, 29305, 32766, 27942, 21622, 31373, 22971, 32670, 32503, 32745, 
#                   29710, 32768, 28226, 30339, 32686, 30386, 31640, 31445, 30960, 29628, 28111, 15704, 14898, 
#                   12883, 7907, 6698, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# 筛选后的特征大小列表
feature_sizes = [139, 5, 638, 5234, 6, 3, 24, 26, 19, 12, 49, 19, 57, 35, 26, 4, 4, 3, 4, 2, 2, 2, 2, 2]
# feature_sizes = [139, 5, 5234, 6, 3, 24, 26, 12, 49, 19, 57, 35, 26, 4, 4, 3, 2, 2, 2]
# feature_sizes = [139, 5234, 24, 26, 12, 49, 19, 57, 35, 26, 4, 4, 3]
# feature_sizes = [139, 5234, 26, 12, 49, 19, 57, 35, 26]
# feature_sizes = [139, 5234, 26, 12, 49, 57, 26]

# 类别特征进行重新划分
# feature_sizes = [31896, 21732, 32685, 32761, 2822, 3628, 21666, 31801, 22472, 21956, 29427, 30386, 31640, 
                #  31445, 30960, 29628, 28111, 15704, 4, 2, 2, 2, 2, 2]

# 将数值特征也当作类别特征处理
# feature_sizes = [139, 5, 638, 5234, 6, 3, 24, 26, 19, 12, 49, 19, 57, 35, 26, 4, 4, 3, 4, 2, 2, 
#                  2, 2, 2, 8882, 27, 20, 34, 1828, 389, 220, 516, 1810, 1593, 1718, 465, 379, 11, 13]

# 表示数值特征从第42列开始
# continous_features = 41
continous_features = 24
# continous_features = 19
# continous_features = 13
# continous_features = 9
# continous_features = 39

# click_feature 代表预测的是click -True 还是 install -False
click_feature = False

# alldata_feature 代表读取的数据是第一个csv还是全部csv all_data -True
alldata_feature = True

# lab_id表示实验的Id编号，用于保存图片、结果、方便记录


