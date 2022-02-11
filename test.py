import numpy as np
import cv2
import fluidsim

s = fluidsim.Simulation((300,100),1.7)

cv2.namedWindow("Sim",cv2.WINDOW_KEEPRATIO)
#debug = "Directional Densities"
#cv2.namedWindow(debug,cv2.WINDOW_KEEPRATIO)

#s.densities[5,0,125:175] = 1

#s.densities[4] = np.random.random(s.shape)*.1+.9

stepping = True
step = False
stream = False

#s.densities[3,100:200,150:155] = .1
#s.densities[5,100:200,140:145] = .1

source = np.zeros(s.shape,bool)
source[:5,10:90] = True

s.sources.append((source,np.array([.2,0])))

xx,yy = np.meshgrid(np.arange(0,s.shape[1]),np.arange(0,s.shape[0]))
xx-=int(s.shape[1]/2)
yy-=int(s.shape[0]/4)
s.walls[(xx<yy)&(xx>-25)&(xx<25)&(yy>-25)&(yy<25)] = True
#s.walls[(xx**2+yy**2)**.5<15] = True
s.walls[:,0]=True
s.walls[:,-1]=True

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('output.mp4',fourcc, 120.0, s.shape, True)

skip = 5
counter = 0
smok=False

source = np.zeros(s.shape,bool)
source[int(s.shape[1]/2)+50,25] = True
source[int(s.shape[1]/2)+50,50+24] = True
dye = fluidsim.Smoke(s.shape,source)

while 1:

    if (not stepping) or (stepping and step):
        if stream:
            s.streaming()
            stream = not stream
        else:
            #print(ma,mi)
            s.collision()
            #print(d.max(),d.min())
            stream = not stream
        step=False

        #vel = s.velocity()
        #mod = (vel**2).sum(axis=2)**.5
        #show = mod*(-dye._alignment(vel))
        #show = show.T
        #cv2.imshow(debug,(show+1)/2)
        dye.step(s.velocity(),smok)
    d = np.zeros(s.shape)
    #d = s.density()
    d = (s.velocity()**2).sum(axis=2)
    
    ma,mi = (d.max(),d.min())
    #print(s.densities.max())
    d = d/d.max() if d.max()!=0 else d
    d *=.2
    d[s.walls] = 1
    
    show = (d.T*255)[:,:,np.newaxis].repeat(3,axis=2)
    dyef = dye.field
    dyef /= dyef.max() if dyef.max()>0 else 1
    dyef = np.log(dyef+.3)-np.log(.3)
    dyef /= dyef.max() if dyef.max()>0 else 1
    show[:,:,1] = show[:,:,1] + ((dyef).T*255)
    show[show>255]=255
    show[show<0]=0
    show = show.astype(np.uint8)
    if counter==0:
        print("BOB")
        out.write(show)
    counter+=1
    counter%=skip

    cv2.imshow("Sim",show)

    # img = np.zeros((d.shape[1]*3,d.shape[0]*3))
    # for i in range(9):
    #     x = i%3
    #     y = i//3
    #     img[y*d.shape[1]:(y+1)*d.shape[1],x*d.shape[0]:(x+1)*d.shape[0]] = s.densities[i].T/s.densities[i].T.max()
    # cv2.imshow(debug,img)

    k = cv2.waitKey(1)&0xff
    if k==ord('q'):
        break
    if k==ord('p'):
        stepping = not stepping
    if k==ord('s'):
        step=True
    if k==ord('d'):
        smok=not smok