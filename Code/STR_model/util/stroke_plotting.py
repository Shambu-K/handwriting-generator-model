# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_strokes(word_strokes) -> list:
    ''' Get strokes from a word with multiple strokes.'''
    word_strokes[-1, -1] = 1  # Set the EoS flag of the last point to 1
    strokes = []
    start = -1
    for cur in range(word_strokes.shape[0]):
        if word_strokes[cur, -2] == 1:
            start = cur
        if word_strokes[cur, -1] == 1 and start != -1:
            strokes.append(word_strokes[start:cur+1])
            start = -1
            
    return strokes


def plot_word_strokes(word_strokes, color='black', title='', split_strokes=True):
    ''' Plot the strokes of a word.'''
    if split_strokes: strokes = get_strokes(word_strokes)
    else: strokes = [word_strokes]
    for stroke in strokes:
        plt.plot(stroke[:, 0], -stroke[:, 1], color=color)
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.show()


def plot_str_word_strokes(strokes, color='black', title='', split_strokes=True):
    ''' Plot the directions of the strokes of a word.'''
    # Create a figure
    fig, ax = plt.subplots()
    plt.title(title)
    x_min, x_max = np.min(strokes[:, 0]), np.max(strokes[:, 0])
    y_min, y_max = np.min(strokes[:, 1]), np.max(strokes[:, 1])

    # Set the axis limits
    ax.set_xlim(x_min - 10, x_max + 10)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Plot the SoS and EoS points
    SoS_indices = np.where(strokes[:, -2] == 1)[0]
    EoS_indices = np.where(strokes[:, -1] == 1)[0]
    ax.scatter(strokes[SoS_indices, 0], strokes[SoS_indices, 1], color='green', label='Start of stroke', s=10)
    ax.scatter(strokes[EoS_indices, 0], strokes[EoS_indices, 1], color='red', label='End of stroke', s=10)

    def plot_stroke_segment(ax, strokes, start, end):
        dx = np.diff(strokes[start:end, 0])
        dy = np.diff(strokes[start:end, 1])
        ax.quiver(strokes[start:end-1, 0], strokes[start:end-1, 1], dx, dy, angles='xy', scale_units='xy', scale=1)

    if split_strokes:
        start = 0
        for i in range(len(strokes)):
            if strokes[i, -2] == 1 and i != 0:  # start of a new stroke
                plot_stroke_segment(ax, strokes, start, i)
                start = i
            elif i == len(strokes) - 1:  # end of the last stroke
                plot_stroke_segment(ax, strokes, start, i)
    else:
        plot_stroke_segment(ax, strokes, 0, len(strokes))

    ax.legend()
    plt.show()

def animate_word(word_strokes, color='black', title='', speed=1, save_path=None, split_strokes=True):
    ''' Animate the strokes of a word.'''
    fig, ax = plt.subplots()
    fig.suptitle(title)
    padding = 10
    xlim = (word_strokes[:, 0].min()-padding, word_strokes[:, 0].max()+padding)
    ylim = (word_strokes[:, 1].min()-padding//2, word_strokes[:, 1].max()+padding//2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_aspect('equal')
    
    lines = []
    pen = ax.scatter([], [], color='red')
    start = 0
    
    def animate(i):
        nonlocal start
        if split_strokes and word_strokes[i, 2] == 1 and i != 0:
            start = i
        x = word_strokes[start:i+1, 0]
        y = word_strokes[start:i+1, 1]
        line, = ax.plot(x, y, color=color)
        lines.append(line)
        
        pen.set_offsets([x[-1], y[-1]]) # Move the pen
        return lines + [pen]
    
    ani = animation.FuncAnimation(fig, animate, frames=len(word_strokes), interval=30/speed, blit=True)
    if save_path is not None:
        ani.save(save_path, writer='pillow', fps=30)
    plt.close()
    
def test_plots():
    stroke_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/stroke_5488.npy'
    strokes = np.load(stroke_path)
    strokes = np.delete(strokes, 2, axis=1)
    
    plot_word_strokes(strokes)
    plot_str_word_strokes(strokes)
    animate_word(strokes, speed=1, save_path='./test.gif')
    
if __name__ == '__main__':
    test_plots()
    
# %%