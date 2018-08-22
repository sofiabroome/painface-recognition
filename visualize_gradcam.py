import numpy as np
import cv2

from skimage.transform import resize
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

def visualize(image, conv_output, conv_grad, gb_viz, number):
    image = image[0]
    if len(gb_viz.shape) > 3:
        gb_viz = gb_viz[0]
    output = conv_output[0]           # [7,7,512]
    grads_val = conv_grad[0]          # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)
    weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32) # [7,7]
    

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (128,128), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')
    plt.savefig('gradcam_results/gradcam_' + str(number) + '_input.png')
   
    from PIL import Image 
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    bg = Image.fromarray((255*img).astype('uint8'))
    overlay = Image.fromarray(cam_heatmap.astype('uint8'))
    blend = Image.blend(bg, overlay, 0.2)
    imgplot = plt.imshow(blend)
    ax.set_title('Input Image with GradCAM Overlay')
    plt.savefig('gradcam_results/gradcam_' + str(number) + '_overlay.png')

    fig = plt.figure(figsize=(20, 20))    
    ax = fig.add_subplot(131)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')
    
    gb_viz = np.dstack((
            gb_viz[:, :, 0],
            gb_viz[:, :, 1],
            gb_viz[:, :, 2],
        ))       
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = fig.add_subplot(132)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')
    

    gd_gb = np.dstack((
            gb_viz[:, :, 0] * cam,
            gb_viz[:, :, 1] * cam,
            gb_viz[:, :, 2] * cam,
        ))            
    ax = fig.add_subplot(133)
    imgplot = plt.imshow(gd_gb)
    ax.set_title('guided Grad-CAM')

    plt.savefig('gradcam_results/gradcam_' + str(number) + '.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')
    plt.savefig('gradcam_results/gradcam_' + str(number) + '_guidedbp.png')

    
    
