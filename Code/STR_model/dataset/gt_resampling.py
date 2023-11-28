# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from util.stroke_plotting import get_strokes

def interpolate_stroke(points: np.ndarray, m: int, num_EoS_extra: int=5) -> list[np.ndarray]:
    ''' Interpolate points along a single stroke.
    
    Parameters:
    - points: n x (x, y, SoS, EoS)
    - m: number of equidistant points to output
    - num_EoS_extra: number of extra EoS points to add to help with class imbalance during training
    
    Returns: m x (x, y, SoS, EoS)
    '''
    # Calculate the cumulative distance along the stroke
    cumulative_distance = np.cumsum(np.sqrt(np.diff(points[:, 0])**2 + np.diff(points[:, 1])**2))
    cumulative_distance = np.insert(cumulative_distance, 0, 0)  # Insert 0 at the beginning
    
    # Normalize the cumulative distance to range [0, 1]
    if cumulative_distance[-1] == 0:
        normalized_distance = np.zeros_like(cumulative_distance)
    else:
        normalized_distance = cumulative_distance / cumulative_distance[-1]
    
    # Interpolate equidistant points
    interpolated_normalized_distance = np.linspace(0, 1, max(1, m-num_EoS_extra))
    interpolated_points = np.zeros((m, 4))
    interpolated_points[:len(interpolated_normalized_distance), 0] = np.interp(interpolated_normalized_distance, normalized_distance, points[:, 0])
    interpolated_points[:len(interpolated_normalized_distance), 1] = np.interp(interpolated_normalized_distance, normalized_distance, points[:, 1])
    
    # Add the last point num_EoS_extra times
    interpolated_points[max(1, m-num_EoS_extra):] = points[-1]
    
    # Set the SoS flag
    interpolated_points[0, 2] = 1
    
    return interpolated_points.tolist()


def resample_strokes(word_strokes: np.ndarray, m: int,  num_EoS_extra: int=10) -> np.ndarray:
    ''' Interpolate points along a word with multiple strokes to get m equidistant points describing the same strokes.
    
    Parameters:
    - word_strokes: list of n x (x, y, SoS, EoS)
    - m: number of equidistant points to output
    - num_EoS_extra: number of extra EoS points to add to help with class imbalance during training
    
    Returns: m x (x, y, SoS, EoS)
    '''
    interpolated_word_strokes = []
    strokes = get_strokes(word_strokes)
    n = word_strokes.shape[0]
    m_cur = 0
    for i, stroke in enumerate(strokes):
        if i < len(strokes)-1: 
            stroke_len = (m * len(stroke)) // n
            m_cur += stroke_len
        else:
            stroke_len = m - m_cur
        if stroke_len == 0: continue
        interpolated_word_strokes += interpolate_stroke(stroke, stroke_len, num_EoS_extra)
    assert len(interpolated_word_strokes) == m, f'{len(interpolated_word_strokes)=} is not equal to {m=}'
        
    return np.array(interpolated_word_strokes)
    
def draw_interpolated_strokes(strokes: np.ndarray, interpolated_strokes: np.ndarray) -> None:
    ''' Draw the original and interpolated strokes.'''
    # Plot the original points and the interpolated stroke
    plt.scatter(strokes[:,0], strokes[:,1])
    print(f'Num of strokes: {len(get_strokes(strokes))}')
    for interpolate_stroke in get_strokes(interpolated_strokes):
        plt.plot(interpolate_stroke[:,0], interpolate_stroke[:,1], color='red')
        # plt.scatter(interpolate_stroke[:,0], interpolate_stroke[:,1], color='red')
    plt.legend(['Original points', 'Interpolated stroke'])
    plt.show()
    
# %%
def test_gt_resampling() -> None:
    ''' Test the ground truth resampling.'''
    file_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/stroke_2.npy'
    points = np.load(file_path)
    points = np.delete(points, 2, axis=1) # Remove the 3rd dimension (time) from the points (Keep 0, 1, 3, 4)

    print(f'{points.shape=}')

    # Set the number of equidistant points for interpolation
    img_width = max(points[:, 0])
    num_equidistant_points = int(img_width)
    print(f'{int(img_width)=}, {num_equidistant_points=}')

    # Get the interpolated points
    interpolated_points = resample_strokes(points, num_equidistant_points)

    draw_interpolated_strokes(points, interpolated_points)
    
if __name__ == '__main__':
    test_gt_resampling()