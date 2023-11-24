# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_strokes(word_strokes) -> list:
    ''' Get strokes from a word with multiple strokes.'''
    word_strokes[-1, 3] = 1  # Set the EoS flag of the last point to 1
    strokes = []
    start = -1
    for cur in range(word_strokes.shape[0]):
        if word_strokes[cur, 2] == 1:
            start = cur
        if word_strokes[cur, 3] == 1 and start != -1:
            strokes.append(word_strokes[start:cur+1])
            start = -1
            
    return strokes


def plot_word_strokes(word_strokes, color='black', title=''):
    ''' Plot the strokes of a word.'''
    strokes = get_strokes(word_strokes)
    for stroke in strokes:
        plt.plot(stroke[:, 0], -stroke[:, 1], color=color)
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.show()

    
def plot_str_word_strokes(strokes, color='black', title=''):
    ''' Plot the directions of the strokes of a word.'''
        # Create a figure
    fig, ax = plt.subplots(figsize=(6, 4))

    x_min, x_max = np.min(strokes[:, 0]), np.max(strokes[:, 0])
    y_min, y_max = np.min(strokes[:, 1]), np.max(strokes[:, 1])

    # Set the axis limits
    ax.set_xlim(x_min - 50, x_max + 50)
    ax.set_ylim(y_min - 50, y_max + 50)

    # Invert y-axis
    ax.invert_yaxis()

    start = 0
    for i in range(len(strokes)):
        if strokes[i, 2] == 1 and i != 0:  # start of a new stroke
            dx = np.diff(strokes[start:i, 0])
            dy = np.diff(strokes[start:i, 1])
            ax.quiver(strokes[start:i-1, 0], strokes[start:i-1, 1], dx, dy, angles='xy', scale_units='xy', scale=1)
            ax.scatter(*strokes[start, :2], color='green', s=10)  # start of stroke
            ax.scatter(*strokes[i-1, :2], color='red', s=10)  # end of stroke
            start = i
        elif i == len(strokes) - 1:  # end of the last stroke
            dx = np.diff(strokes[start:, 0])
            dy = np.diff(strokes[start:, 1])
            ax.quiver(strokes[start:i, 0], strokes[start:i, 1], dx, dy, angles='xy', scale_units='xy', scale=1)
            ax.scatter(*strokes[start, :2], color='green', s=10)  # start of stroke
            ax.scatter(*strokes[i, :2], color='red', s=10)  # end of stroke

    # Assign labels for the legend
    ax.scatter([], [], color='green', label='Start of stroke')
    ax.scatter([], [], color='red', label='End of stroke')

    ax.legend()
    plt.show()


def animate_word(word_strokes, color='black', title='', speed=1, save_path=None):
    ''' Animate the strokes of a word.'''
    fig, ax = plt.subplots()    
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
        if word_strokes[i, 2] == 1 and i != 0:
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
    plt.show()
    
    
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