

def get_config():
    return {
        'S' : 2/3,
        'A' : 1.7159,
        'efficient' : True,
        'train_path' : './data/train',
        'test_path' : './data/test',
        'num_epochs' : 1,
        'batch_size' : 1,
        'test_batch_size' : 64,
        'test_num_batches' : 50,
        'j' : 3,
        'lr' : [5e-4, 5e-4, 2e-4, 2e-4, 2e-4, 5e-5, 5e-5, 5e-5, 5e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
        'damping' : 0.2,
        'mu' : 0.99,
        'latest_path' : './checkpoints/run_'
    }