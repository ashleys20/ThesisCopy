import albumentations as A
import glob
import cv2
from enum import Enum


class AugmentationTypes(Enum):
    COLORJITTER = "colorjitter"
    DOWNSCALE = "downscale"
    FLIPHORIZ = "fliphoriz"
    FLIPVERT = "flipvert"
    FLIPBOTH = "flipboth"
    GAUSSBLUR = "gaussblur"
    GAUSSNOISE = "gaussnoise"
    INVERT = "invert"
    MOTIONBLUR = "motionblur"
    RANDOMSHADOW = "randomshadow"
    RANDOMSUNFLARE = "randomsunflare"
    RANDOMTONECURVE = "randomtonecurve"

def choose_aug_type(aug):
    if aug is AugmentationTypes.COLORJITTER:
        transform = A.Compose([A.ColorJitter(p=1.0)])
    elif aug is AugmentationTypes.DOWNSCALE:
        transform = A.Compose([A.Downscale(scale_min=0.25, scale_max=0.25, p=1.0)])
    elif aug is AugmentationTypes.FLIPHORIZ:
        transform = A.Compose([A.HorizontalFlip(p=1.0)])
    elif aug is AugmentationTypes.FLIPVERT:
        transform = A.Compose([A.VerticalFlip(p=1.0)])
    elif aug is AugmentationTypes.FLIPBOTH:
        transform = A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])
    elif aug is AugmentationTypes.GAUSSBLUR:
        transform = A.Compose([A.GaussianBlur(p=1.0)])
    elif aug is AugmentationTypes.GAUSSNOISE:
        transform = A.Compose([A.GaussNoise(p=1.0)])
    elif aug is AugmentationTypes.INVERT:
        transform = A.Compose([A.InvertImg(p=1.0)])
    elif aug is AugmentationTypes.MOTIONBLUR:
        transform = A.Compose([A.MotionBlur(p=1.0)])
    elif aug is AugmentationTypes.RANDOMSHADOW:
        transform = A.Compose([A.RandomShadow(p=1.0)])
    elif aug is AugmentationTypes.RANDOMSUNFLARE:
        transform = A.Compose([A.RandomSunFlare(p=1.0)])
    elif aug is AugmentationTypes.RANDOMTONECURVE:
        transform = A.Compose([A.RandomToneCurve(p=1.0)])
    return transform


def augment_data(isLeft):
    #if isLeft is True: folder =  '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/Thesis_Data_Videos_Left/*'
    #else: folder = '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/Thesis_Data_Videos_Right/*'
    folder = '/Users/ashley20/PycharmProjects/ThesisCameraCalibration/OrigFrames/*'


    #loop through every video in traning folder
    #loop through every frame in orgimages folder
    for filepath in glob.glob(folder):
        # read video
        #video = cv2.VideoCapture(filepath)
        #success, frame = video.read()
        #count = 0
        #if isLeft:
         #   video_name = filepath[80:]
        #else:
         #   video_name = filepath[81:]

        #print(video_name)

        #loop through every frame in video
        #while success:
        #count += 1
        #frame_name = video_name + "_" + str(count) + "_orig" + ".jpg"
        #converting frame from bgr to rgb for albumentations compatibility
        #cv2.imwrite('OrigFrames/'+ frame_name, frame)
        frame = cv2.imread(filepath)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #for every frame, perform each augmentation and save that frame
        for aug in AugmentationTypes:
            #print(aug)
            val = aug.value
            transform = choose_aug_type(aug)
            transformed = transform(image=frame)
            transformed_image = transformed["image"]
            transformed_image_name = filepath[67:-8] + val + ".jpg"
            #print(transformed_image_name)
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            #cv2.imshow('transformed image', transformed_image)
            #cv2.waitKey(0)
            cv2.imwrite('AugmentedFrames/' + transformed_image_name, transformed_image)

        #go to next frame
        #success, frame = video.read()


if __name__ == '__main__':
   print("in data augmentation")
   isLeft = True
   augment_data(isLeft)
   #isLeft = False
   #augment_data(isLeft)
