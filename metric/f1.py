def f1_score(y_true:list, y_pred:list):
    '''
    precision = len(intersection(y_true, y_pred)) / len(y_pred)
    recall = len(intersection(y_true, y_pred)) / len(y_true)
    f1 = (2 * (precision * recall)) / (precision + recall) => (2 * len(intersection(y_true, y_pred))) / (len(y_true) + len(y_pred))
    '''
    
    intersection = list(set(y_true) & set(y_pred))
    
    return (2 * len(intersection)) / float(len(y_true) + len(y_pred))
