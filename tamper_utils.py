import wave

def insert_segment(orig_seg, insert_seg, time, save_path, L):
    """
    Insert an L second segment at a given time in the original query
    
    Args:
        orig_seg (str): path to original audio segment (query)
        insert_seg (bytes?): frames to insert
        time (int): time in frames to insert the segment to
        save_path (str): path to save the resulting 10+L second long audio
    """

    frames = []

    wave1 = wave.open(orig_seg, 'rb')
    frames.append([wave1.getparams(),wave1.readframes(wave1.getnframes())])
    wave1.close()

    frames.append([None,insert_seg])

    part1 = frames[0][1][:time]
    part2 = frames[0][1][time:]
    insert = frames[1][1]

    result = wave.open(save_path,'wb')
    result.setparams(frames[0][0])
    result.writeframes(part1)
    result.writeframes(insert)
    result.writeframes(part2)
    result.close()
    
    
def delete_segment(orig_seg, time, save_path, L):
    """
    Delete an L second segment from the original query starting at given time
    (L in frames)
    Args:
        orig_seg (str): path to original audio segment (query)
        time (int): start time in frames to delete a L sec segment from
        save_path (str): path to save the resulting 10+L second long audio
    """
    frames = []

    wave1 = wave.open(orig_seg, 'rb')
    frames.append([wave1.getparams(),wave1.readframes(wave1.getnframes())])
    wave1.close()

    part1 = frames[0][1][:time]
    part2 = frames[0][1][time + L:]

    result = wave.open(save_path,'wb')
    result.setparams(frames[0][0])
    result.writeframes(part1)
    result.writeframes(part2)
    result.close()

    
def replace_segment(orig_seg, replace_seg, time, save_path, L):
    """
    Replace an L secong segment with given segment in the original query starting at given time
    (L in frames)
    Args:
        orig_seg (str): path to original audio segment (query)
        replace_seg (bytes?): frames to replace with
        time (int): time in ms to start replacement at
        save_path (str): path to save the resulting 10 second long audio
    """

    frames = []

    wave1 = wave.open(orig_seg, 'rb')
    frames.append([wave1.getparams(),wave1.readframes(wave1.getnframes())])
    wave1.close()

    frames.append([None,replace_seg])

    part1 = frames[0][1][:time]
    part2 = frames[0][1][time + L:]
    replace = frames[1][1]

    result = wave.open(save_path,'wb')
    result.setparams(frames[0][0])
    result.writeframes(part1)
    result.writeframes(replace)
    result.writeframes(part2)
    result.close()
    
