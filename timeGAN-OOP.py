"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks


- Updated the timegan.py file from the original TimeGAN github project to be compatiable with tf2

-----------------------------

timeGAN-OOP.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from my_utils import extract_time, rnn_cell, random_generator, batch_generator


def timegan(ori_data, parameters):
    """TimeGAN function.
    
    Use original data as training set to generate synthetic data (time-series)
    
    Args:
        - ori_data: original time-series data (already pre-processed)
        - parameters: TimeGAN network parameters (in the form of a Python dictionary)
        
    Returns:
        - generated_data: generated time-series data
    """

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)
    
    def MinMaxScaler(data):
        """Min-Max Normalizer.
        Args:
            - data: raw data
      
        Returns:
            - norm_data: normalized data
            - min_val: minimum values (for renormalization)
            - max_val: maximum values (for renormalization)
      """     
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    # Define model inputs
    X = tf.keras.Input(shape=(max_seq_len, dim), name="myinput_x")
    Z = tf.keras.Input(shape=(max_seq_len, z_dim), name="myinput_z")
    T = tf.keras.Input(shape=(1,), dtype=tf.int32, name="myinput_t")
    
    # Define Models
    class Embedder(tf.keras.Model):
        """Embedding network between original feature space to latent space.
    
    Args:
      - X: input time-series features
      - T: input time information
      
    Returns:
      - H: embeddings
    """
        def __init__(self):
            super(Embedder, self).__init__()
            self.rnn_cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            self.multi_rnn_cell = tf.keras.layers.StackedRNNCells(self.rnn_cells)
            self.rnn_layer = tf.keras.layers.RNN(self.multi_rnn_cell, return_sequences=True, return_state=False)
            self.dense = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

        def call(self, X, T):
            e_outputs = self.rnn_layer(X, mask=tf.sequence_mask(T))
            H = self.dense(e_outputs)
            return H

    class Recovery(tf.keras.Model):
        """Recovery network from latent space to original space.
        
        Args:
          - H: latent representation
          - T: input time information
          
        Returns:
          - X_tilde: recovered data
        """            
        def __init__(self):
            super(Recovery, self).__init__()
            self.rnn_cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            self.multi_rnn_cell = tf.keras.layers.StackedRNNCells(self.rnn_cells)
            self.rnn_layer = tf.keras.layers.RNN(self.multi_rnn_cell, return_sequences=True, return_state=False)
            self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

        def call(self, H, T):
            r_outputs = self.rnn_layer(H, mask=tf.sequence_mask(T))
            X_tilde = self.dense(r_outputs)
            return X_tilde

    class Generator(tf.keras.Model):
        """Generator function: Generate time-series data in latent space.
            
            Args:
              - Z: random variables
              - T: input time information
              
            Returns:
              - E: generated embedding
            """        
        def __init__(self):
            super(Generator, self).__init__()
            self.rnn_cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            self.multi_rnn_cell = tf.keras.layers.StackedRNNCells(self.rnn_cells)
            self.rnn_layer = tf.keras.layers.RNN(self.multi_rnn_cell, return_sequences=True, return_state=False)
            self.dense = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

        def call(self, Z, T):
            g_outputs = self.rnn_layer(Z, mask=tf.sequence_mask(T))
            E = self.dense(g_outputs)
            return E

    class Supervisor(tf.keras.Model):
        """Generate next sequence using the previous sequence.
   
   Args:
     - H: latent representation
     - T: input time information
     
   Returns:
     - S: generated sequence based on the latent representations generated by the generator
   """
        def __init__(self):
            super(Supervisor, self).__init__()
            self.rnn_cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)]
            self.multi_rnn_cell = tf.keras.layers.StackedRNNCells(self.rnn_cells)
            self.rnn_layer = tf.keras.layers.RNN(self.multi_rnn_cell, return_sequences=True, return_state=False)
            self.dense = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

        def call(self, H, T):
            s_outputs = self.rnn_layer(H, mask=tf.sequence_mask(T))
            S = self.dense(s_outputs)
            return S

    class Discriminator(tf.keras.Model):
        """Discriminate the original and synthetic time-series data.
  
  Args:
    - H: latent representation
    - T: input time information
    
  Returns:
    - Y_hat: classification results between original and synthetic time-series
  """        
        def __init__(self):
            super(Discriminator, self).__init__()
            self.rnn_cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
            self.multi_rnn_cell = tf.keras.layers.StackedRNNCells(self.rnn_cells)
            self.rnn_layer = tf.keras.layers.RNN(self.multi_rnn_cell, return_sequences=True, return_state=False)
            self.dense = tf.keras.layers.Dense(1, activation=None)

        def call(self, H, T):
            d_outputs = self.rnn_layer(H, mask=tf.sequence_mask(T))
            Y_hat = self.dense(d_outputs)
            return Y_hat

    # Instantiate models
    embedder = Embedder()
    recovery = Recovery()
    generator = Generator()
    supervisor = Supervisor()
    discriminator = Discriminator()

    # Optimizers
    optimizer = tf.keras.optimizers.Adam()
    E_optimizer = tf.keras.optimizers.Adam()
    D_optimizer = tf.keras.optimizers.Adam()
    G_optimizer = tf.keras.optimizers.Adam()
    GS_optimizer = tf.keras.optimizers.Adam()


    # Define loss functions
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    # Begin Training
    
    @tf.function
    def train_step_embedder(X_mb, T_mb):
        with tf.GradientTape() as tape:
            H = embedder(X_mb, T_mb)
            X_tilde = recovery(H, T_mb)
            E_loss_T0 = mse_loss(X_mb, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)
        gradients = tape.gradient(E_loss0, embedder.trainable_variables + recovery.trainable_variables)
        E_optimizer.apply_gradients(zip(gradients, embedder.trainable_variables + recovery.trainable_variables))
        return tf.sqrt(E_loss_T0)

    @tf.function
    def train_step_supervisor(X_mb, Z_mb, T_mb):
        with tf.GradientTape() as tape:
            E_hat = generator(Z_mb, T_mb)
            H_hat = supervisor(E_hat, T_mb)
            H_hat_supervise = supervisor(embedder(X_mb, T_mb), T_mb)
            G_loss_S = mse_loss(embedder(X_mb, T_mb)[:, 1:, :], H_hat_supervise[:, :-1, :])
        gradients = tape.gradient(G_loss_S, generator.trainable_variables + supervisor.trainable_variables)
        GS_optimizer.apply_gradients(zip(gradients, generator.trainable_variables + supervisor.trainable_variables))
        return G_loss_S

    @tf.function
    def train_step_joint(X_mb, Z_mb, T_mb):
        for _ in range(2):
            with tf.GradientTape() as tape:
                E_hat = generator(Z_mb, T_mb)
                H_hat = supervisor(E_hat, T_mb)
                X_hat = recovery(H_hat, T_mb)
                Y_fake = discriminator(H_hat, T_mb)
                Y_real = discriminator(embedder(X_mb, T_mb), T_mb)
                Y_fake_e = discriminator(E_hat, T_mb)
                D_loss_real = bce_loss(tf.ones_like(Y_real), Y_real)
                D_loss_fake = bce_loss(tf.zeros_like(Y_fake), Y_fake)
                D_loss_fake_e = bce_loss(tf.zeros_like(Y_fake_e), Y_fake_e)
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
                G_loss_U = bce_loss(tf.ones_like(Y_fake), Y_fake)
                G_loss_U_e = bce_loss(tf.ones_like(Y_fake_e), Y_fake_e)
                G_loss_S = mse_loss(embedder(X_mb, T_mb)[:, 1:, :], supervisor(embedder(X_mb, T_mb), T_mb)[:, :-1, :])
                G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X_mb, [0])[1] + 1e-6)))
                G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X_mb, [0])[0])))
                G_loss_V = G_loss_V1 + G_loss_V2
                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
                E_loss_T0 = mse_loss(X_mb, recovery(embedder(X_mb, T_mb), T_mb))
                E_loss0 = 10 * tf.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1 * G_loss_S
            gradients = tape.gradient(G_loss, generator.trainable_variables + supervisor.trainable_variables)
            G_optimizer.apply_gradients(zip(gradients, generator.trainable_variables + supervisor.trainable_variables))
            gradients = tape.gradient(D_loss, discriminator.trainable_variables)
            D_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
            gradients = tape.gradient(E_loss, embedder.trainable_variables + recovery.trainable_variables)
            E_optimizer.apply_gradients(zip(gradients, embedder.trainable_variables + recovery.trainable_variables))

    @tf.function
    def generate_data(Z_mb, T_mb):
        E_hat = generator(Z_mb, T_mb)
        H_hat = supervisor(E_hat, T_mb)
        X_hat = recovery(H_hat, T_mb)
        return X_hat

    # Training the embedder network
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        step_e_loss = train_step_embedder(X_mb, T_mb)

    # Training the supervisor network
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        step_s_loss = train_step_supervisor(X_mb, Z_mb, T_mb)

    # Joint Training
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        train_step_joint(X_mb, Z_mb, T_mb)

    # Synthetic data generation
    generated_data = []
    for _ in range(no // batch_size + 1):
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        X_hat_curr = generate_data(Z_mb, T_mb)
        generated_data.append(X_hat_curr.numpy())

    generated_data = np.concatenate(generated_data, axis=0)
    generated_data = generated_data[:no]
    generated_data = generated_data * (max_val + 1e-7) + min_val

    return generated_data  
   
   
   
   
   
   
   
   
   
   
   
   