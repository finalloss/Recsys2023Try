# 预先求得的类别特征信息
feature_sizes = [23, 139, 5, 638, 6, 5234, 1, 6, 7, 3, 24, 26, 331, 19, 5854, 12, 49, 924, 19, 57, 35, 
                 26, 4, 4, 3, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# 表示数值特征从第42列开始
continous_features = 41

# click_feature 代表预测的是click -True 还是 install -False
click_feature = False

# alldata_feature 代表读取的数据是第一个csv还是全部csv all_data -True
alldata_feature = True

# 方便调参而定义的类，说明有哪些参数可以调整
class Model_parameters():
    def __init__(self):
        self.lr = 1e-4