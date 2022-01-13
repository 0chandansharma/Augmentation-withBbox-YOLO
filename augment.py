import imgaug as ia
import imgaug.augmenters as iaa
import os
import cv2
import numpy as np
from util import sequence

anno_INPUT_DIR = r'annotation_files/'
img_INPUT_DIR = r'image_files/'
OUTPUT_DIR = 'output_dir/'
AUGMENT_SIZE = 6


def main():
    for file_name in os.listdir(anno_INPUT_DIR):
        annotation_path = anno_INPUT_DIR + file_name



        if os.path.exists(img_INPUT_DIR + file_name.strip('.txt')+'.jpg'):
            image_path = img_INPUT_DIR  + file_name.strip('.txt') + '.jpg'
        else:
            image_path = img_INPUT_DIR + file_name.strip('.txt') + '.png'
        try:
            print(annotation_path)
            augment(image_path,annotation_path,file_name)
        except:
            print("error while reading data")



def augment(image_path, annotation_path,file_name):
    seq = sequence.get()
    for i in range(AUGMENT_SIZE):
        sp = file_name.split('.')
        outfile = '%s/%s-%02d.%s' % (OUTPUT_DIR, sp[0], i, sp[-1])
        file = open(outfile,'w')

        seq_det = seq.to_deterministic()
        _bbs = []
        try:
            image = cv2.imread(image_path)
            annotation = open(annotation_path,'r')
            im_height, im_width, im_channels = image.shape
            lines = annotation.readlines()
            for line in lines:
                values = line.strip('\n').split(',')
                n_x, n_y, n_w, n_h = (values[1], values[2], values[3], values[4])
                label= values[0]
                h = int(float(n_h) * im_height)
                w = int(float(n_w) * im_width)
                x = int((float(n_x) * im_width) - w / 2)
                y = int((float(n_y) * im_height) - h / 2)
                bb = ia.BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h, label=label)
                _bbs.append(bb)
                # cv2.rectangle(image, (int(float(x)), int(float(y))), (int(float(x+w)), int(float(y+h))),
                #               (1, 255, 234),
                #               4)
            bbs = ia.BoundingBoxesOnImage(_bbs, shape=image.shape)

            # seqe = iaa.Sequential([
            #     iaa.AdditiveGaussianNoise(scale=0.05 * 255),
            #     iaa.Affine(translate_px={"x": (1, 5)})
            # ])
            # image_aug, bbs_aug = seqe(images=image, bounding_boxes=bbs)

            image_aug = seq_det.augment_images([image])[0]
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0].remove_out_of_image().cut_out_of_image()

            # writer = Writer(outfile,
            #                 im_width,
            #                 im_height)
            res = ''
            for bb in bbs_aug.bounding_boxes:
                if int((bb.x2-bb.x1)*(bb.y2-bb.y1)) == 0:
                    print("augmentet boundingbox has non existing area. Skipping")
                    continue
                x_norm= (bb.x1 + (bb.x2 - bb.x1)/2)/im_width
                y_norm= (bb.y1 + (bb.y2 - bb.y1)/2)/im_height
                w_norm = (bb.x2 - bb.x1)/im_width
                h_norm = (bb.y2 - bb.y1)/im_height
                # res += str(bb.label)+' '+ str("{:.5f}".format(x_norm))+' '+str("{:.5f}".format(y_norm))+' '+str("{:.5f}".format(w_norm))+ ' ' +str("{:.5f}".format(h_norm)) + '\n'

                if bb.label=='a':
                    leb=0
                elif bb.label=='b':
                    leb=1
                elif bb.label=='c':
                    leb=2
                elif bb.label=='d':
                    leb=3
                else:
                    print(bb.label)
                    leb=0
                # res += str(leb)+' '+ str(x_norm)+' '+str(y_norm)+' '+str(w_norm)+ ' ' +str(h_norm) + '\n'
                res += str(leb) + ' ' + str("{:.5f}".format(x_norm)) + ' ' + str("{:.5f}".format(y_norm)) + ' ' + str("{:.5f}".format(w_norm)) + ' ' + str("{:.5f}".format(h_norm)) + '\n'
                # print(res)
                # print(file)
                # cv2.rectangle(image_aug.astype(np.int32), (int(float(bb.x1)), int(float(bb.y1))), (int(float(bb.x2)), int(float(bb.y2))), (36, 255, 12),
                #               2)
            # cv2.imshow("resin",image)

            # cv2.imshow("resout",image_aug)
            # cv2.waitKey(0)
            # cv2.imwrite('sample_in.jpg',image)
            # cv2.imwrite('sample_ou.jpg',image_aug)
            print(file_name)
            print(res)
            file.write(res)
            cv2.imwrite(outfile.replace('.txt','.jpg'), image_aug)

            file.close()
        except Exception as e:
            print(e)
            cv2.imshow("test",image_aug)
            cv2.waitKey(0)
            print("testing")


if __name__ == '__main__':
    main()
