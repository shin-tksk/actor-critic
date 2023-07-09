import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('3_2movie_cartpole.gif')  # 動画を保存

def huber_loss(x,y,delta):
    dis = abs(x-y)
    if dis <= delta:
        return (dis**2) / 2
    else:
        return (delta * dis) - (delta**2) / 2
