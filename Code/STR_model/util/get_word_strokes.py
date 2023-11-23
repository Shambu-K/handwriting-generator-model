import numpy as np

def get_strokes(word_strokes: np.ndarray) -> list[np.ndarray]:
    ''' Get strokes from a word with multiple strokes.'''
    word_strokes[-1, 3] = 1  # Set the EoS flag of the last point to 1
    strokes = []
    SoS_indices = np.where(word_strokes[:, 2] == 1)[0]
    EoS_indices = np.zeros(len(SoS_indices), dtype=int)
    for i in range(len(SoS_indices)):
        # Find the first EoS after the current SoS (There can be multiple EoS per SoS)
        EoS_indices[i] = np.where(word_strokes[SoS_indices[i]:, 3] == 1)[0][0] + SoS_indices[i]
    
    for i, j in zip(SoS_indices, EoS_indices):
        strokes.append(word_strokes[i:j+1])
    return strokes