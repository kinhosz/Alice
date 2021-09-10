from PIL.Image import init
import Alice
import tkinter as tk
import random
import numpy as np
import matplotlib.pyplot as plt

def isOk(predict,l):
	
	pred = -1
	ans = -0.1

	for i in range(len(l)):
		if l[i] > ans:
			ans = l[i]
			pred = i

	return pred == predict

def read_image_files(url):
	f = open(url,"rb")
	magic_number = int.from_bytes(f.read(4), byteorder="big")
	number_of_images = int.from_bytes(f.read(4), byteorder="big")
	rows = int.from_bytes(f.read(4),byteorder="big")
	columns = int.from_bytes(f.read(4),byteorder="big")

	images = []

	for img in range(number_of_images):
		image = []
		for i in range(rows):
			for j in range(columns):
				pixel = int.from_bytes(f.read(1),byteorder="big",signed=False)
				image.append(pixel)
		images.append(image)

	f.close()

	return images

def read_label_files(url):
	f = open(url,"rb")
	magic_number = int.from_bytes(f.read(4), byteorder="big")
	number_of_items = int.from_bytes(f.read(4), byteorder="big")

	labels = []

	for lbl in range(number_of_items):
		label = int.from_bytes(f.read(1),byteorder="big",signed=False)
		labels.append(label)

	f.close()

	return labels

def setPixel(data, x, y, large):

    for i in range(x,x+large):
        if i >= 300:
            break
        for j in range(y, y+large):
            if j >= 400:
                break
            data[i][j] = [255, 0,0]

def fill(data):

    sz_x = len(data)
    sz_y = len(data[0])

    for j in range(sz_y):
        for i in range(sz_x):
            if data[i][j] == [255,0,0]:
                break
            data[i][j] = [0,0,0]

def graph(L):

    width, height = 400,300

    data = []
    
    for i in range(height):
        row = []
        for j in range(width):
            row.append([125,0,0])
        data.append(row)

    if len(L) == 1:
        space = width
    else:
        space = width/(len(L) - 1)

    initial = 0
    sz = len(L)

    last_w = 0
    last_h = height - 1

    abs_pos = 0

    for i in range(sz):
        coef = L[i]
        pos = int(height - height*coef)
        pos = max(pos, 0)
        pos = min(pos, height-1)

        for j in range(last_w, initial):
            setPixel(data, last_h, j, 2)
        
        for j in range(min(last_h,pos), max(last_h,pos)):
            setPixel(data,  j, initial, 2)

        last_h = pos
        last_w = initial

        abs_pos = abs_pos + space
        initial = int(abs_pos)
        initial = min(initial, width-1)

    fill(data)

    return np.asarray(data).astype("uint8")

def render(page: Alice.Page):

    # get the parent reference from the window
    parent = page.parent()
    # shape of window
    WIDTH, HEIGHT = parent.shape()
    # style of page
    pallete = parent.getPallete()
    # get the text style
    text = parent.getText()
    # buffer for events
    buffer = Alice.Queue()
    # handle responde
    handle = Alice.HandleEvent(buffer)

    # create a gui workspace
    canvas = tk.Canvas(parent, width=WIDTH, height=HEIGHT, bg=pallete["meta"])
    canvas.grid(columnspan=1, rowspan=3)

    # create the text1_content of the page
    text1_content = tk.StringVar()
    text1_label = tk.Label(textvariable=text1_content, bg=pallete["meta"])
    text1_label.grid(column=0, row=0, columnspan=1, rowspan=1)
    text1_label.config(font=(text["font"], text["h2"], text["style"]))
    text1_label.config(fg=pallete["border"])
    text1_content.set("Loading ... 0%")

    # create the neural network
    robot = Alice.Network([28*28,150,75,10],eta = 0.2)

    # training the robot

    images = read_image_files("dataset/train-images.idx3-ubyte")
    labels = read_label_files("dataset/train-labels.idx1-ubyte")
    
    hit = 0
    epoch_size = 1000
    test = 0
    epoch = 0
    eixo_x = []
    eixo_y = []
    
    out = []
    for i in range(10):
        out.append(0.0)

    repet = 3
    size_data = 60000
        
    for lazy in range(repet):
        train_test = [(x,y) for x, y in zip(images,labels)]
        random.shuffle(train_test)
        images = [x for (x,y) in train_test]
        labels = [y for (x,y) in train_test]
        for i in range(size_data):
            
            ans = robot.send(images[i])
            if isOk(labels[i],ans):
                hit = hit + 1
                
            test = test + 1
            out[labels[i]] = 1.0
            robot.learn(images[i],out)
            out[labels[i]] = 0.0
            
            if test == epoch_size:
                rate = hit/epoch_size
                epoch = epoch + 1
                test = 0
                hit = 0
                eixo_y.append(rate)
                # convert to graphic
                data = graph(eixo_y)
                img = Alice.Picture(data)
                graph_label = tk.Label(image=img.image(), background=pallete["border"])
                graph_label.grid(column=0, row=1, columnspan=1, rowspan=2,padx=20, pady=20)
                # update text
                perc = lazy*size_data + i
                perc = (perc*100)//(repet*size_data)
                text1_content.set("Loading ... " + str(perc) + "%")
                if parent.flip() == False:
                    break

    return False
