import numpy as np
import glob

label_files = sorted(glob.glob("./data/train/*seg.npy"))

total_voxels = 0
zero_voxels = 0
one_voxels = 0
two_voxels = 0
three_voxels = 0

largest_dims = [0, 0, 0]
smallest_dims = [300, 300, 300]

for label_file in label_files:
    mat = np.load(label_file)

    total_voxels += np.prod(np.shape(mat))
    zero_voxels += np.count_nonzero(mat == 0)
    one_voxels += np.count_nonzero(mat == 1)
    two_voxels += np.count_nonzero(mat == 2)
    three_voxels += np.count_nonzero(mat == 3)


    for i in range(len(np.shape(mat))):
        #print(np.shape(mat))
        if np.shape(mat)[i] > largest_dims[i]:
            largest_dims[i] = np.shape(mat)[i]

        if np.shape(mat)[i] < smallest_dims[i]:
            smallest_dims[i] = np.shape(mat)[i]


print(total_voxels)
print(zero_voxels / zero_voxels)
print(zero_voxels / one_voxels)
print(zero_voxels / two_voxels)
print(zero_voxels / three_voxels)

print(largest_dims)
print(smallest_dims)
# print(data[0][100][100][100])
# img = Image.fromarray(200 * data[0][100], "F")

# height_list = []
# width_list = []
# depth_list = []
#
# for height_val in range(np.shape(data)[1]):
#     height_list.append(Image.fromarray(200 * data[0,height_val,:,:], "F"))
#
# for width_val in range(np.shape(data)[2]):
#     width_list.append(Image.fromarray(200 * data[0,:,width_val,:], "F"))
#
# for depth_val in range(np.shape(data)[3]):
#     depth_list.append(Image.fromarray(200 * data[0,:,:,depth_val], "F"))
#
# img.save("height.gif", save_all=True, append_images=height_list)
# img.save("width.gif", save_all=True, append_images=width_list)
# img.save("depth.gif", save_all=True, append_images=depth_list)
