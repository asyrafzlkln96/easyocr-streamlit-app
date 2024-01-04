import pandas as pd
import numpy as np
import streamlit as st
import easyocr
import PIL
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def display_image(file):
	image = Image.open(file) # read image with PIL library
	st.image(image) #display image

	# it will only detect the English and Turkish part of the image as text
	reader = easyocr.Reader(['ms','en'], gpu=False) 
	result = reader.readtext(np.array(image))  # turn image to numpy array
	return result

def rectangle(image, result):
    # https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
    """ draw rectangles on image based on predicted coordinates"""
    draw = ImageDraw.Draw(image)
    for res in result:
        top_left = tuple(res[0][0]) # top left coordinates as tuple
        bottom_right = tuple(res[0][2]) # bottom right coordinates as tuple
        draw.rectangle((top_left, bottom_right), outline="blue", width=2)
    #display image on streamlit
    st.image(image)


# main title
st.title("Get text from image with EasyOCR") 
# subtitle
st.markdown("## EasyOCR with Streamlit(English and Malay)")

# upload image file
file = st.file_uploader(label = "Upload your image", type=['png', 'jpg', 'jpeg'])

result = display_image(file)

textdic_easyocr = {} 
for idx in range(len(result)): 
  pred_coor = result[idx][0] 
  pred_text = result[idx][1] 
  pred_confidence = result[idx][2] 
  textdic_easyocr[pred_text] = {} 
  textdic_easyocr[pred_text]['pred_confidence'] = pred_confidence

 # create a dataframe which shows the predicted text and prediction confidence
  df = pd.DataFrame.from_dict(textdic_easyocr).T
  st.table(df)

rectangle(image, result)