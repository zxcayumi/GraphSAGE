device = 'cuda'

def trans_view( features, dimension):
    'transform features from flatten to dimension shape'
    return features.view(dimension)