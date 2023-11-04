print("Importing...")


from scipy import ndimage
import glob
import cv2
import os

print("Imported...")

folder = "data/ALL_IDB2/t_aug/"
save_path = "data/ALL_IDB2/t_aug_noise/"
filetype = "*.tif"
images = []
t_img = []
image_paths = glob.glob(os.path.join(folder, filetype))


data_len = len(image_paths)
print("Total images: ", data_len)

blr = 12
blr_level = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
# lb_level=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75]
# hb_level=lb_level
hb = 15
lb = 20
# lc_level=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
# hc_level=lc_level
lc = hc = 10

image_name = [x[-11:] for x in image_paths]
i = 0
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))

    # img=img/255

    if img is not None:
        for j in range(max(blr, hb, lb, hc, lc)):
            if blr > j:
                x = blr_level[j]
                blr_img = ndimage.uniform_filter(img, size=(x, x, 1))
                blr_img = blr_img.astype("uint8")
                cv2.imwrite(save_path + "BLR" + str(j + 1) + "_" + filename, blr_img)

            if hb > j:
                hb_img = img + (5 * (j + 1))
                hb_img = (
                    hb_img / hb_img.max()
                ) * 255  # normalization instead of clipping
                hb_img = hb_img.astype("uint8")
                cv2.imwrite(save_path + "HB" + str(j + 1) + "_" + filename, hb_img)

            if lb > j:
                lb_img = img - (2 * (j + 1))
                # lb_img=(lb_img/lb_img.max())*255         #normalization instead of clipping
                lb_img = lb_img.astype("uint8")
                cv2.imwrite(save_path + "LB" + str(j + 1) + "_" + filename, lb_img)

            if hc > j:
                hc_img = img * (1 + (j + 1) / 10)
                # hc_img=(hc_img/hc_img.max())*255         #normalization instead of clipping
                hc_img = hc_img.astype("uint8")
                cv2.imwrite(save_path + "HC" + str(j + 1) + "_" + filename, hc_img)

            if lc > j:
                lc_img = img / (1 + (j + 1) / 10)
                # lc_img=(lc_img/lc_img.max())*255        #normalization instead of clipping
                lc_img = lc_img.astype("uint8")
                cv2.imwrite(save_path + "LC" + str(j + 1) + "_" + filename, lc_img)
            # j=j+1

        i = i + 1
        print(i, "--", filename)
