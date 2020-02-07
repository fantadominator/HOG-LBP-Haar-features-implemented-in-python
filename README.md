# HOG-LBP-Haar-features-implemented-in-python
Using numpy to implement typical  features of Digital Image Processing, including HOG, LBP, Haar
1.
For HOG features:

hog_descriptor = HOG(img,block_size=3)
hog_vector = hog_descriptor.hog_features()
print(hog_vector.shape)


2.
For LBP features:

lbp = LBP(img)

features = lbp.extend_lbp(3,8)  # features = lbp.original_lbp()

r_features = lbp.extend_lbp(3,8,rotation_sensitive=False)

vector = lbp.get_lbp_vector(features,8)

print(vector.shape,vector[:10])


3.
For basic Haar features:

haar_features = Haar(img)

vector = haar_features.get_haar_features()


The Haar features may take more time to calculate.
