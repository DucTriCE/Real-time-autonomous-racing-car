import cv2
import os

images = os.listdir('./sign_more_data')

for i in images:
    img = cv2.imread(f'./sign_more_data/{i}')
    img = cv2.resize(img,(320,320))
    cv2.imwrite(f'./sign_more_data/{i}',img)
print('Done!!!')

