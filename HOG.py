import skimage
from skimage import io,color,filters
import matplotlib.pyplot as plt
import numpy as np
img = io.imread('dog1.jpg')
img = color.rgb2gray(img)
x = filters.sobel_h(img)
y = filters.sobel_v(img)
magnitude = np.sqrt(x**2+y**2)
plt.imshow(magnitude,plt.cm.gray)
print(img.shape)
plt.show()


def L2_norm_1d(vector:np.ndarray):
    norm = np.sum(np.square(vector))
    vector /=np.sqrt(norm+1e-7)
    return vector


class HOG:
    def __init__(self,img,cell_size=8,bin_size=9,block_size=2):
        self.img = img
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.block_size = block_size

    def hog_features(self):
        magnitude,angle = self.global_gradient()

        cell_num_x = self.img.shape[1]//self.cell_size
        cell_num_y = self.img.shape[0]//self.cell_size

        cell_matrix = []
        for y in range(cell_num_y):
            for x in range(cell_num_x):
                cell_magnitude = magnitude[y*self.cell_size:(y+1)*self.cell_size,x*self.cell_size:(x+1)*self.cell_size]
                cell_angle     = angle[y*self.cell_size:(y+1)*self.cell_size,x*self.cell_size:(x+1)*self.cell_size]
                cell_vector = self.cal_cell_hog(cell_magnitude,cell_angle)
                cell_matrix.append(cell_vector)

        cell_matrix = np.array(cell_matrix).reshape([cell_num_y,cell_num_x,self.bin_size])
        print(cell_matrix.shape)
        #block
        hog_vector=[]
        step_x = cell_matrix.shape[1]-self.block_size+1
        step_y = cell_matrix.shape[0]-self.block_size+1
        for i in range(step_y):
            for j in range(step_x):
                block_vector=[]
                block_vector.extend(cell_matrix[i][j])
                block_vector.extend(cell_matrix[i][j+1])
                block_vector.extend(cell_matrix[i+1][j])
                block_vector.extend(cell_matrix[i+1][j+1])
                block_vector = L2_norm_1d(np.array(block_vector))
                hog_vector.append(block_vector)

        return np.array(hog_vector).reshape(-1)

    # return magnitude,angle

    def global_gradient(self):
        gradient_x = filters.sobel_h(self.img)
        gradient_y = filters.sobel_v(self.img)

        magnitude = np.sqrt(gradient_x**2+gradient_y**2)
        angle = np.arctan(gradient_y/(gradient_x+1e-5))
        angle[angle >= 180] -= 180
        return magnitude, angle

    def cal_cell_hog(self, cell_mag,cell_ang):
        cell_histogram = np.zeros(self.bin_size)
        w = self.cell_size
        bin_angle = 180//self.bin_size
        for i in range(w):
            for j in range(w):
                ang = cell_ang[i,j]
                mag = cell_mag[i,j]
                min_angle = int(bin_angle*(ang//bin_angle))
                max_angle = min_angle+bin_angle
                left_ratio = (ang-min_angle)/bin_angle
                right_ratio = (max_angle - ang)/bin_angle
                left_mag = mag*left_ratio
                right_mag = mag*right_ratio
                cell_histogram[min_angle//bin_angle]+=left_mag
                cell_histogram[(min_angle//bin_angle+1)%9]+=right_mag

        return cell_histogram


'''
hog_descriptor = HOG(img,block_size=3)
hog_vector = hog_descriptor.hog_features()
print(hog_vector.shape)
'''