import numpy as np


def L2_Norm_1d(vector:np.ndarray):
    norm = np.sum(np.square(vector))
    vector /=np.sqrt(norm+1e-7)
    return vector


class Haar:
    def __init__(self,img):
        assert len(img.shape) == 2
        self.img = img

    def cal_integral_graph(self):
        img = self.img
        graph = np.zeros(shape=img.shape)
        W = img.shape[1]
        H = img.shape[0]

        # sum the lines
        for y in range(H):
            for x in range(W):
                if x==0:
                    graph[y,x]  = img[y,x]
                else:
                    graph[y,x] = graph[y,x-1] + img[y,x]

        # sum the colomns
        for x in range(W):
            for y in range(1,H):
                graph[y,x] = graph[y-1,x] + graph[y,x]

        return graph

    def get_haar_features(self):
        haar_vector = []
        W = self.img.shape[1]
        H = self.img.shape[0]
        modes = ([-1,1],[[-1],[1]],[1,-1,1],[[1],[-1],[1]])
        for mode in modes:
            mode = np.array(mode)
            if len(mode.shape) == 1:
                mode = np.repeat(mode[np.newaxis,:],1,axis=0)
            w = mode.shape[1]
            h = mode.shape[0]
            num_extend_x = W//w
            num_extend_y = H//h
            for i in range(1,num_extend_y+1):
                h_t = i*h
                for j in range(1,num_extend_x+1):
                    vector = []
                    w_t = j*w
                    if w == 2:
                        plus_x_min = 0
                        plus_x_max = w_t//2 - 1
                        plus_y_min = 0
                        plus_y_max = h_t - 1
                        weight = 2
                    elif h == 2:
                        plus_x_min = 0
                        plus_x_max = w_t-1
                        plus_y_min = h_t // 2
                        plus_y_max = h_t - 1
                        weight =2
                    elif w == 3:
                        plus_x_min = w_t//3
                        plus_x_max = 2*plus_x_min - 1
                        plus_y_min = 0
                        plus_y_max = h_t-1
                        weight = 3
                    elif h == 3:
                        plus_x_min = 0
                        plus_x_max = w_t - 1
                        plus_y_min = h_t//3
                        plus_y_max = 2*plus_y_min - 1
                    # search the entire image
                    for y in range(H-h_t+1):
                        for x in range(W-w_t+1):
                            total_sum = self.cal_rectangle_sum(x,x+w_t-1,y,y+h_t-1)
                            minus_sum = self.cal_rectangle_sum(x+plus_x_min,x+plus_x_max,y+plus_y_min,y+plus_y_max)
                            value = total_sum - weight*minus_sum
                            vector.append(value)
                    vector = L2_Norm_1d(np.array(vector))
                    haar_vector.extend(vector)



        return np.array(haar_vector)

    def cal_rectangle_sum(self,x_min,x_max,y_min,y_max):
        graph = self.cal_integral_graph()
        if x_min == 0 and y_min == 0:
            value = graph[y_max,x_max]

        elif x_min == 0 and y_min > 0:
            value = graph[y_max,x_max] - graph[y_min-1,x_max]

        elif x_min != 0 and y_min == 0:
            value = graph[y_max,x_max] - graph[y_max,x_min-1]

        else:
            value = graph[y_max,x_max] - graph[y_max,x_min-1] - graph[y_min-1,x_max] + graph[y_min-1,x_min-1]

        return value


# test
'''
img = np.array([[1,3,2,10,5],[5,7,12,1,3],[4,2,3,5,12]])
haar_features = Haar(img)

vector = haar_features.get_haar_features()
print(vector.shape,vector[:10])
'''









