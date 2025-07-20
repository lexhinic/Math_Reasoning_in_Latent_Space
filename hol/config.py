class Config:
    # Data parameters
    HOL_LIGHT_PATH = "./hol_light"
    HOLIST_DB_PATH = "./hol"
    TRAIN_SPLIT = 10 #11655
    VAL_SPLIT = 10 #668
    TEST_SPLIT = 10 #3620
    TIMEOUT = 10  # Timeout for rewrite operations (seconds)
    USE_SIMULATED_HOLLIGHTREWRITER = True  # Use simulated HOLLightRewriter instead of HOLLightReWriter
    
    # Model parameters
    NODE_DIM = 128       # Dimension of node representations in GNN
    EMBED_DIM = 1024     # Dimension of the embedding space
    NUM_HOPS = 16        # Number of message passing hops in GNN
    HIDDEN_DIM = 1024    # Dimension of hidden layers in MLPs
    
    # Training parameters
    BATCH_SIZE = 2     # Number of theorems per batch
    NEG_EXAMPLES = 15    # Number of negative examples per theorem
    LEARNING_RATE = 1e-4 # Learning rate for optimizer
    NOISE_STD = 1e-3     # Standard deviation of noise added to embeddings
    NUM_EPOCHS = 1     # Number of training epochs
    
    # Evaluation parameters
    MAX_REWRITE_STEPS = 1  # Maximum number of rewrite steps for evaluation
