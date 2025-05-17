Amazon Earthworks Detector

Hello!

This project uses artificial intelligence (AI) to find ancient earthworks in 
the Amazon rainforest. Earthworks are human-made shapes like mounds, patterns, 
or raised fields that were built by people long ago. The project works by looking 
at elevation data, which shows the shape of the land, to spot these features.

The area focused on is the Guaporé River Basin in Brazil. Recent 
technology like LiDAR and satellite images have discovered many of 
these hidden earthworks there. This project helps archaeologists by 
automatically detecting these features from the elevation data.

The system works in a few steps: first, it creates fake (synthetic) 
elevation maps with and without earthworks to train the AI. Then, 
it processes these maps to highlight important details like slopes 
and smooths out the data to reduce noise. After that, it trains a 
computer model called a convolutional neural network (CNN) to tell 
if an earthwork is present or not. Finally, it can analyze new elevation 
maps and say whether they likely contain earthworks.

The project includes code to generate training data, train the model, 
and test it on new elevation tiles. You can use it by running simple 
commands in the terminal. It uses Python programming and some common 
libraries like PyTorch, NumPy, and Rasterio.

In the future, this project could be improved by using real elevation 
data collected from satellites and LiDAR scanners. It could also be 
expanded to locate exactly where earthworks are on the map instead of 
just saying if they exist or not. There’s also potential to build a 
user-friendly app to make it easier for researchers to use.

]
