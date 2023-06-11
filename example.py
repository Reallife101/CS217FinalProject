import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from camera import Camera
import structure
import processor
import features

# Download images from http://www.robots.ox.ac.uk/~vgg/data/data-mview.html
'''
def house():
    input_path = 'imgs/house/'
    camera_filepath = 'imgs/house/3D/house.00{0}.P'

    cameras = [Camera(processor.read_matrix(camera_filepath.format(i)))
            for i in range(9)]
    [c.factor() for c in cameras]

    points3d = processor.read_matrix(input_path + '3D/house.p3d').T  # 3 x n
    points4d = np.vstack((points3d, np.ones(points3d.shape[1])))  # 4 x n
    points2d = [processor.read_matrix(
        input_path + '2D/house.00' + str(i) + '.corners') for i in range(9)]

    index1 = 2
    index2 = 4
    img1 = cv2.imread(input_path + 'house.00' + str(index1) + '.pgm')  # left image
    img2 = cv2.imread(input_path + 'house.00' + str(index2) + '.pgm')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(points3d[0], points3d[1], points3d[2], 'b.')
    # ax.set_xlabel('x axis')
    # ax.set_ylabel('y axis')
    # ax.set_zlabel('z axis')
    # ax.view_init(elev=135, azim=90)

    # x = cameras[index2].project(points4d)
    # plt.figure()
    # plt.plot(x[0], x[1], 'b.')
    # plt.show()

    corner_indexes = processor.read_matrix(
        input_path + '2D/house.nview-corners', np.int)
    corner_indexes1 = corner_indexes[:, index1]
    corner_indexes2 = corner_indexes[:, index2]
    intersect_indexes = np.intersect1d(np.nonzero(
        [corner_indexes1 != -1]), np.nonzero([corner_indexes2 != -1]))
    corner_indexes1 = corner_indexes1[intersect_indexes]
    corner_indexes2 = corner_indexes2[intersect_indexes]
    points1 = processor.cart2hom(points2d[index1][corner_indexes1].T)
    points2 = processor.cart2hom(points2d[index2][corner_indexes2].T)

    height, width, ch = img1.shape
    intrinsic = np.array([  # for imgs/house
        [2362.12, 0, width / 2],
        [0, 2366.12, height / 2],
        [0, 0, 1]])

    return points1, points2, intrinsic
'''

def dino(path1,path2):
    # Dino
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    pts1, pts2 = features.find_correspondence_points(img1, img2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].plot(points1[0], points1[1], 'r.')
    ax[1].autoscale_view('tight')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].plot(points2[0], points2[1], 'r.')
    fig.show()
    
    height, width, ch = img1.shape
    intrinsic = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])

    return points1, points2, intrinsic

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def start(path1,path2):
    points1, points2, intrinsic = dino(path1,path2)

    # Calculate essential matrix with 2d points.
    # Result will be up to a scale
    # First, normalize points
    points1n = np.dot(np.linalg.inv(intrinsic), points1)
    points2n = np.dot(np.linalg.inv(intrinsic), points2)
    E = structure.compute_essential_normalized(points1n, points2n)
    print('Computed essential matrix:', (-E / E[0][1]))

    # Given we are at camera 1, calculate the parameters for camera 2
    # Using the essential matrix returns 4 possible camera paramters
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = structure.compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = structure.reconstruct_one_point(
            points1n[:, 0], points2n[:, 0], P1, P2)

        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    #tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
    tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)
    '''
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.add_subplot(projection='3d')
    #ax = fig.gca(projection='3d')
    ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    '''
    return tripoints3d, points1, points2

p3_1, p21_1,p22_1 = start('imgs/viff.003.ppm','imgs/viff.001.ppm')
p3_2, p21_2,p22_2 = start('imgs/viff.005.ppm','imgs/viff.001.ppm')
pt2 = list()
list_pts = list()
for i in range(len(p22_1[0])):
    pts = [p22_1[0][i],p22_1[1][i]]
    pt2.append(pts)
for i in range(len(p22_2[0])):
    pts = [p22_2[0][i],p22_2[1][i]]
    if pts in pt2:
        list_pts.append(pts)
        #print("find")
#if len(list_pts)<3: #should be >2
ind_p1 = list()
ind_p2 = list()
for pts in list_pts:
    for i in range(len(p22_1[0])):
        if(pts[0]==p22_1[0][i] and pts[1]==p22_1[1][i]):
            ind_p1.append(i)
            break
    for i in range(len(p22_2[0])):
        if(pts[0]==p22_2[0][i] and pts[1]==p22_2[1][i]):
            ind_p2.append(i)
            break

#print(len(ind_p1))
#print(len(ind_p2))

pt3_a = [list(),list(),list()]
pt3_b = [list(),list(),list()]

for i in ind_p1:
    pt3_a[0].append(p3_1[0][i])
    pt3_a[1].append(p3_1[1][i])
    pt3_a[2].append(p3_1[2][i])
for i in ind_p2:
    pt3_b[0].append(p3_2[0][i])
    pt3_b[1].append(p3_2[1][i])
    pt3_b[2].append(p3_2[2][i])

R,t = rigid_transform_3D(np.array(pt3_a), np.array(pt3_b))

pt31 = [list(),list(),list()]
pt3 = [list(),list(),list()]
for i in range(len(p3_1[0])):
    pt31[0].append(p3_1[0][i])
    pt31[1].append(p3_1[1][i])
    pt31[2].append(p3_1[2][i])
for i in range(len(p3_2[0])):
    pt3[0].append(p3_2[0][i])
    pt3[1].append(p3_2[1][i])
    pt3[2].append(p3_2[2][i])

newpt3 = R@pt31+t
#print(newpt3)

for i in range(len(newpt3[0])):
    pt3[0].append(newpt3[0][i])
    pt3[1].append(newpt3[1][i])
    pt3[2].append(newpt3[2][i])

fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.add_subplot(projection='3d')
#ax = fig.gca(projection='3d')
ax.plot(pt3[0], pt3[1], pt3[2], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()