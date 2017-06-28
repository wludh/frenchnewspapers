classes = [2, 3, 1, 2, 3, 1, 1, 1, 1]

for i in range(0, len(classes)):
    if classes[i] == 1:
        classes[i] = 'r'
    elif classes[i] == 2:
        classes[i] = '#eeefff'
    elif classes[i] ==3:
        classes[i] = '#eeefff'

print (classes)
