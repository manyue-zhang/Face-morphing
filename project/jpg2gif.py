import imageio.v2 as imageio

gif_images = []

for i in range(60):
    gif_images.append(imageio.imread("D:/Desktop/Face-Morphing/Face-Morphing/results/result/a1_a2_"+str(i)+".jpg"))   # 读取多张图片
imageio.mimsave("facemorphing.gif", gif_images, fps=15)