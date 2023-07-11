import cv2
from PIL import Image

img = cv2.imread("/home/cjm/DeepMaterial/results/001_RADN_archived_20230612_110342/visualization/areaDataset/svbrdf-001_RADN-0.png")
normal = img[0:256, 256:256*2, :]
roughness = img[256:265*2, 256*3:256*4, :]

# cv2.imwrite("/home/cjm/DeepMaterial/tests/data/pnoise_normal.png",normal[141:145, 104:108, :])
cv2.imwrite("/home/cjm/DeepMaterial/tests/data/normal.png", normal)
cv2.imwrite("/home/cjm/DeepMaterial/tests/data/noise_roughness.png", roughness)
