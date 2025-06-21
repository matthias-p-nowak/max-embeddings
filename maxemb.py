import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

embeddings=[]
angles={}

def create_random_vectors(n, edim):
    return [np.random.randn(edim).tolist() for _ in range(n)]

def normalize(vector):
    return vector / np.linalg.norm(vector)  

def spreadout(i,j,alpha):
    # print(f"Spreading out {i} and {j}, alpha: {alpha}")
    ei = embeddings[i]
    ej = embeddings[j]
    dot=np.dot(ei, ej)
    embeddings[i]=normalize(ei - alpha * dot *ej)
    embeddings[j]=normalize( ej-alpha * dot *ei)
    newAngles=[]
    for idx in angles.keys():
        if idx[0]==i or idx[1]==i or idx[0]==j or idx[1]==j:
            o=angles[(idx[0],idx[1])]
            dot=np.abs( np.dot(embeddings[idx[0]], embeddings[idx[1]]))
            angles[(idx[0],idx[1])]=dot 
            # print (f"{idx[0]}/{idx[1]}: {o} -> {dot} {np.arccos(dot)*180/np.pi}")

def plotAdjusted(adjusted):
    res=[]
    for c,v in angles.items():
        dot=np.dot(embeddings[c[0]], embeddings[c[1]])
        a = np.arccos(dot)*180/np.pi
        res.append(a)
    sorted_res = sorted(res)
    adjusted.set_xdata(sorted_res)

def main():
    global embeddings, angles
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <dimension> <number> <iterations>")
        sys.exit(1)
    edim = int(sys.argv[1])
    enumber = int(sys.argv[2])
    iterations= int(sys.argv[3])
    if enumber <= edim:
        print("Error: enumber must be greater than edim")
        sys.exit(1)
    print(f"edim: {edim} number: {enumber} iterations: {iterations}")
    embeddings = create_random_vectors(int(enumber), int(edim))
    embeddings = [normalize(vector) for vector in embeddings]
    # print(embeddings)
    original_angles=[]
    for i in range(len(embeddings)):
        for j in range(i):
            dot=np.dot(embeddings[i], embeddings[j])
            angle=np.arccos(dot)*180/np.pi
            original_angles.append(angle)
            dot=np.abs(dot)
            angle=np.arccos(dot)*180/np.pi
            # print (f"{i}/{j}: {dot} {angle}")
            angles[(i,j)]  = dot
    original_angles.sort()
    plt.plot(original_angles, range(len(original_angles)), label='original')
    adjusted,=plt.plot(original_angles, range(len(original_angles)), label='adjusted')
    plt.xlabel("Angle")
    plt.ylabel("cumulative count")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)
    decay=0.5
    curmin=1000
    run=0
    last_time=datetime.now()
    for k in range(iterations):
        max_angle = max(angles, key=angles.get) 
        if angles[max_angle] < curmin:
            curmin=angles[max_angle]
            run=0
        else:
            run+=1
        if run> 50:
            if decay < 1e-20:
                break
            decay*=0.9
            run=0
            print(f"##### Decay: {decay:10.5e} #####")
            pass
        # print(f"Max angle: {k:8} {angles[max_angle]:10.5f} {np.arccos(angles[max_angle])*180/np.pi:8.3f} {decay:8.4e}")
        spreadout(max_angle[0], max_angle[1], decay)
        time_diff=datetime.now()-last_time
        if time_diff.total_seconds() > 2:
            last_time=datetime.now()
            plotAdjusted(adjusted)
            plt.pause(0.1)
    plotAdjusted(adjusted)
    plt.show()
    print("Done")

if __name__ == "__main__":
    main()