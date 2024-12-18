

def get_config():
    return {
        'S' : 2/3,
        'A' : 1.7159,
        
        # maintaing a list of connections where 
        # a list corresponds to the filter of its 
        # index number's incoming channels
        'C3' : [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 0],
            [5, 0, 1],
            
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 0],
            [4, 5, 0, 1],
            [5, 0, 1, 2],
            
            [5, 0, 1, 2],
            
        ]
    }