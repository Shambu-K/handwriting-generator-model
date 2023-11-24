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