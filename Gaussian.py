from skimage import filters
from skimage.util import img_as_ubyte
import pandas as pd
import numpy as np

#reading in the csv file
data = pd.read_csv('x_train_gr_smpl.csv')

#setting up empty dataframe
newdata = pd.DataFrame()

#performing gaussian filter on each row, and ensuring data is in correct (0-255) form
for index, row in data.iterrows():
    image = row.to_numpy(dtype=np.uint8).reshape((48,48))
    edges = img_as_ubyte(filters.gaussian(image, sigma=0.4))
    array = edges.reshape((1,2304))
    df = pd.DataFrame.from_dict(array)
    newdata = newdata.append(df,ignore_index=True)

#output data to a new csv file
newdata.to_csv("gaussian_x_train_gr_smpl.csv", index=False, encoding='utf8')

#reduced output data to a new csv file
newdata.to_csv("gaussian_x_train_gr_smplREDUCED.csv", columns =(705,706,740,847,895,940,941,1094,1141,1142,1172,1178,1223,1224,1225,1271,1272,1273,1274,1319,1320,1321,1322,1369,1370,1470,1471,1506,1507,1518,1519,1520,1553,1554,1555,1695,1713,1714,1743,1752,1849), index=False, encoding='utf8')

