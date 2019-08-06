import numpy as np
def readAdjMatrix(path):
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    first_ele = True
    for data in f.readlines():
        ## 去掉每行的换行符，"\n"
        data = data.strip('\n')
        ## 按照 空格进行分割。
        nums = data.split(" ")
        ## 添加到 matrix 中。
        if first_ele:
            ### 将字符串转化为整型数据
            nums = [int(x) for x in nums ]
            ### 加入到 matrix 中 。
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [int(x) for x in nums]
            matrix = np.c_[matrix,nums]
    matrix = matrix.transpose()
    f.close()	
    return matrix

# test.
if __name__ == '__main__':
    matrix = readAdjMatrix("matrix")
    print(matrix)