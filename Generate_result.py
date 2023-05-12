
import pandas as pd

test_data = pd.read_csv("../recsys2023_data/test/000000000000.csv",sep='\t')
RowId = test_data.iloc[:,0]
i = 0
with open('./click_result.txt', 'r') as f1, open('./install_result.txt', 'r') as f2, open('./result.txt', 'w') as r:
    r.write("RowId\tis_clicked\tis_installed\n")
    for line1, line2 in zip(f1, f2):
        # 将 line1 和 line2 整合成新的一行
        r.write(str(RowId[i])+'\t')
        r.write(line1.strip() +'\t'+ line2.strip() + '\n')
        i+=1