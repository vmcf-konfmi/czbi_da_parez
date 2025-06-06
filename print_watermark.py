from watermark import watermark

# This script prints the watermark information for the current Python environment.
print(watermark())
print(watermark(packages="cv2,skimage,pandas,numpy,stardist,tensorflow"))