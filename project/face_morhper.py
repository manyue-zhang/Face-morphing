import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pandas as pd

# 1. dlib Verwenden , um die Merkmalspunkte des Gesichts zu erhalten
def get_points(image): 
    predictor_model = 'D:/Desktop/DBV/code/shape_predictor_68_face_landmarks.dat'
    face_detector = dlib.get_frontal_face_detector() 
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print('No face detected in image {}'.format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)  #  landmark/Koordinaten bekommen
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])

    #Vier Eckpunkte + Mittelpunkte von vier Kanten
    x = image.shape[1] - 1
    y = image.shape[0] - 1
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])

    return np.array(points)

#Manuelle Auswahl von Gesichtsschlüsselpunkten
def get_mpoints(img):
    
    plt.imshow(img)
    points = np.array([])
    points = plt.ginput(n=50, timeout=0)
    plt.plot(points, 'ro')
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    tri = Delaunay(points)
    plt.triplot(x, y, tri.simplices.copy())
    points = [np.array([x, y])]
    print(tri.points)
    return points

# Delaunay-Triangulation Verwenden, um Punktsätze zu Dreiecken einer bestimmten Größe zu verbinden
def get_triangles(points):  
    return Delaunay(points).simplices

#Warps und Alpha-Mischungen dreieckiger Regionen von img1 und img2 bis img
def morph_triangle(img1, img2, img, tri1, tri2, tri, alpha):  
    #ein Begrenzungsrechteck für jedes Dreieck finden
    rect1 = cv2.boundingRect(np.float32([tri1]))  # oberen linken Ecke, die Länge, Breite von tri1
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    #Versatzpunkte um die linke obere Ecke der jeweiligen Rechtecke
    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []

    for i in range(0, 3):
        tri_rect_warped.append(
            ((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(
            ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(
            ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

    #warpImage auf kleine rechteckige Blöcke anwenden
    img1_rect = img1[rect1[1]:rect1[1] +
                     rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] +
                     rect2[3], rect2[0]:rect2[0] + rect2[2]]
    size = (rect[2], rect[3])
    warped_img1 = affine_transform(
        img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(
        img2_rect, tri_rect2, tri_rect_warped, size)

    # gewichtete Summe 
    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

    # mask generieren/das Dreieck füllen, um die Maske zu erhalten
    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

    # mask verwenden/
    #Kopiert den dreieckigen Bereich des rechteckigen Blocks in das Ausgabebild
    img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
            rect[2]] * (1 - mask) + img_rect * mask

# affine Transformation auf dem Gesicht durchführen, um die Position zu bestimmen
def affine_transform(input_image, input_triangle, output_triangle, size): 
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle))
    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image

# Bilder morph
def morph_faces(filename1, filename2, alpha=0.5):  
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]),interpolation=cv2.INTER_CUBIC)
    print('img1.shape',img1.shape)
    print('img2.shape',img2.shape)

    points1 = get_points(img1)
    print('pionts1:',len(points1),points1)
    points2 = get_points(img2)
    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
    # Points speichern
    p = pd.DataFrame(points)
    p.to_csv('./points.csv')

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    triangles = get_triangles(points)
    for i in triangles:
        x = i[0]
        y = i[1]
        z = i[2]

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]
        morph_triangle(img1, img2, img_morphed, tri1, tri2, tri, alpha)

    return np.uint8(img_morphed)


def main(file1,file2,alpha):
    try:
        alpha = float(alpha)
    except:
        alpha = 0.5
    img_morphed = morph_faces(file1, file2, alpha)
    output_file = '{}_{}_{}.jpg'.format(
        file1.split('.')[0][-2:], file2.split('.')[0][-2:], alpha)
    cv2.imwrite(output_file, img_morphed)
    return output_file

def main2(file1,file2,alpha):
    alpha = 0
    for i in range(60):
        alpha += 0.0166
        img_morphed = morph_faces(file1, file2, alpha)
        output_file = 'D:/Desktop/Face-Morphing/Face-Morphing/results/result/{}_{}_{}.jpg'.format(
        file1.split('.')[0][-2:], file2.split('.')[0][-2:], i)
        cv2.imwrite(output_file, img_morphed)

if __name__ == '__main__':
    filename1 = "D:\Desktop\Face-Morphing\Face-Morphing\images/aligned_images/mona1.jpg"
    filename2 = "D:\Desktop\Face-Morphing\Face-Morphing\images/aligned_images/mona2.jpg"
    main2(file1=filename1,file2=filename2,alpha=0)
