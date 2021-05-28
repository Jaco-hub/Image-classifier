# Perform the predictions
# Import necessary 
import argparse
from utility_functions import *

# def get_input_args():
parser = argparse.ArgumentParser()
parser.add_argument('--chkp_location', default = 'checkpoint.pth', help = 'Where will the checkpoint be saved?')
parser.add_argument('--image_path', type = str, default = 'flowers/test/16/image_06657.jpg', help = 'Path to image that will be analyzed')
parser.add_argument('--top_k', type = int, default = 3, help = "Top number of most likely cases")
parser.add_argument('--cat_name', default = 'cat_to_name.json', help = 'Name of the file mapping the category numbers to names')
parser.add_argument('--gpu', type = bool, default = True, help = 'Use the GPU or cpu?')
args = parser.parse_args()

# load the model from the checkpoint
model = load_checkpoint(args.chkp_location)
# load the mapping of category numbers to names
cat_to_name = load_mapping(args.cat_name)
# using the GPU?
device = gpu_cpu(args.gpu)
# pre-process the image
image = process_image(args.image_path)
# imshow function to show image
# ax = imshow(image, ax=None, title=None)
# select the top number of likely cases for the picture
#predict(args.image_path, device, model, args.top_k)
#ps_topk, ps_topk_class = predict(args.image_path, device, model, args.top_k)
results = predict(model, args.image_path, device, args.top_k)
# show the results of classifying the picture!
# plot_solution(image, ps_topk, ps_topk_class, model)
# print results
for name, probability in results:
    print('{}: {}%'.format(name,round(float(probability)*100, 1)))