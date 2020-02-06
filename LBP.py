from skimage import io,color,filters
import matplotlib.pyplot as plt
import numpy as np


def L2_Norm_1d(vector:np.ndarray):
    norm = np.sum(np.square(vector))
    vector /=np.sqrt(norm+1e-7)
    return vector


class LBP:
    def __init__(self,img):
        self.img = img
        self.shape = img.shape

    def original_lbp(self):
        feature_map = np.zeros(shape=self.img.shape,dtype=self.img.dtype)

        for y in range(1,self.shape[0]-1):
            for x in range(1,self.shape[1]-1):
                center = self.img[y,x]
                code = 0
                code |= (self.img[y - 1, x - 1] >= center) << np.uint8(7)
                code |= (self.img[y - 1, x] >= center) << np.uint8(6)
                code |= (self.img[y - 1, x + 1] >= center) << np.uint8(5)
                code |= (self.img[y , x - 1] >= center) << np.uint8(4)
                code |= (self.img[y , x + 1] >= center) << np.uint8(3)
                code |= (self.img[y + 1, x - 1] >= center) << np.uint8(2)
                code |= (self.img[y + 1, x ] >= center) << np.uint8(1)
                code |= (self.img[y + 1, x + 1] >= center) << np.uint8(0)
                feature_map[y,x] = code

        return feature_map


    def bilinear_intersection(self,x,y):
        x_min = int(x)
        x_max = x_min + 1
        y_min = int(y)
        y_max = y_min + 1
        if x_max<1 or y_max<1 or x_min>=self.shape[1]-1 or y_min >=self.shape[0]-1:
            return 0
        else:
            Ax = (x-x_min)/(x_max-x+1e-7)
            Ay = (y-y_min)/(y_max-y+1e-7)
            N1 = self.img[y_min,x_min]
            try:
                N2 = self.img[y_min,x_max]
            except:
                print(x_min,x_max,x)
            N3 = self.img[y_max,x_min]
            N4 = self.img[y_max,x_max]
            w1 = 1/((1+Ay)*(1+Ax))
            w2 = Ax * w1
            w3 = Ay * w1
            w4 = Ax * w3
            value = w1*N1 + w2*N2 + w3*N3 + w4*N4
            return value

    def extend_lbp(self,radius=1,p_num=8,rotation_sensitive=True):
        P = p_num
        R = radius

        featrues_map = np.zeros(shape=self.shape,dtype=np.uint8)

        w = self.shape[1]
        h = self.shape[0]
        for y in range(1,h-1):
            for x in range(1,w-1):
                # center greyscale
                center = self.img[y,x]

                code = 0
                for i in range(P):
                    x_i = x + R*np.sin(2*np.pi*i/P)
                    y_i = y - R*np.cos(2*np.pi*i/P)
                    intersect_value = self.bilinear_intersection(x_i,y_i)
                    code |= (intersect_value>=center) << np.uint8(P-1-i)

                if not rotation_sensitive:
                    code = self.rotation_insensitive(code,P)
                featrues_map[y,x] = int(code)

        return featrues_map

    def rotation_insensitive(self,code,P):
        minimum = code
        for i in range(1,P):
            temp = (np.uint8)((code >> i) | (code << P-i))
            if temp < minimum:
                minimum = temp

        return minimum

    def get_lbp_vector(self,feature_map,P,block_num_x=8,block_num_y=8):
        w = feature_map.shape[1]
        h = feature_map.shape[0]
        block_size_x = int(w / block_num_x)
        block_size_y = int(h / block_num_y)
        result = np.zeros(shape=[block_num_x*block_num_y,2**P])
        resultindex=0

        for i in range(block_num_y):
            for j in range(block_num_x):
                hist_block=np.zeros(2**P)
                feature_block = feature_map[i*block_size_y:(i+1)*block_size_y,j*block_size_x:(j+1)*block_size_x].reshape(-1)
                for pixel in feature_block:
                    print(pixel)
                    hist_block[pixel] +=1

                hist_block = L2_Norm_1d(hist_block)
                result[resultindex] = hist_block
                resultindex += 1

        return result.reshape(-1)


'''
img = io.imread('cat1.jpg')
img = color.rgb2gray(img)
print(img.shape)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

lbp = LBP(img)

features = lbp.extend_lbp(3,8)
plt.imshow(features,cmap=plt.cm.gray)
plt.title('extend')
plt.show()
original = lbp.extend_lbp(3,8,rotation_sensitive=False)
plt.imshow(original,cmap=plt.cm.gray)
plt.title('rotation insensitive')
plt.show()

vector = lbp.get_lbp_vector(features,8)
print(vector.shape,vector[:10])
'''



