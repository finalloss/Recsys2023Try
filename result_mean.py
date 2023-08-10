# 读取./result81.txt下的文件，计算第三列的平均值

# 读取txt文件

def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

def get_remean(path_ls):
    path_num = len(path_ls)
    lines_all = []
    rename = "./re_mean"
    for i in range(path_num):
        path_name = "./result" + str(path_ls[i]) + ".txt"
        rename = rename + str(path_ls[i]) + "_"
        lines_all.append(read_txt(path_name))
    rename = rename + ".txt"
    with open(rename,"w") as f:
        f.write(lines_all[0][0])
        for i in range(1,len(lines_all[0])):
            mean_num = 0
            # for j in range(path_num):
            #     line = lines_all[j][i].split("\t")
            #     mean_num += float(line[2])
            # mean_num = mean_num / path_num
            # mean_num = 0.1 * float(lines_all[0][i].split("\t")[2]) + 0.9 * float(lines_all[1][i].split("\t")[2])
            mean_num = 0.4 * float(lines_all[0][i].split("\t")[2]) + 0.4 * float(lines_all[1][i].split("\t")[2]) + 0.2 * float(lines_all[2][i].split("\t")[2])
            f.write(lines_all[0][i].split("\t")[0] + '\t' + lines_all[0][i].split("\t")[1] + '\t' + str(mean_num) + '\n')

# get_remean([40,42,50,55,58,62,76,81,83])
get_remean([2,40,62])