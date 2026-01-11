
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import scipy.sparse as sp

# ==============================
# 1. DATA LOADING AND PREPROCESSING
# ==============================

def load_nci_almanac_data(data_path):
    """Load and preprocess NCI-ALMANAC data"""
    # Mount Google Drive if needed
    """try:
        from google.colab import drive
        drive.mount('/content/Drive/')
    except:
        pass"""

    # Load SMILES data
    all_SMILES = pd.read_csv(data_path + 'drugs__SMILES.csv')
    all_SMILES.set_index('Drug', inplace=True)

    # Load dataset
    all_dataset = pd.read_csv(data_path + 'NCI-ALMANAC_subset_555300.csv')
    print(f"Dataset size: {len(all_dataset)} rows")

    # Create combo_id for grouping
    all_dataset['combo_id'] = all_dataset.apply(
        lambda row: f"{row['Drug1']}_{row['Drug2']}_{row['CellLine']}", axis=1
    )

    # SMILES encoding dictionary (from your code)
    smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                   "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                   "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                   "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                   "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                   "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                   "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                   "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

    # Prepare SMILES sequences for drugs
    MAX_SMI_LEN = 100
    drug_smiles_cache = {}

    for drug in all_SMILES.index:
        smiles = all_SMILES.loc[drug][0]
        encoded = np.zeros(MAX_SMI_LEN)
        for i, ch in enumerate(smiles[:MAX_SMI_LEN]):
            encoded[i] = smiles_dict.get(ch, 0)
        drug_smiles_cache[drug] = encoded

    # Get unique drugs from dataset
    all_drugs = set(all_dataset['Drug1'].unique()) | set(all_dataset['Drug2'].unique())
    for drug in all_drugs:
        if drug not in drug_smiles_cache:
            # Use zero vector for unknown drugs
            drug_smiles_cache[drug] = np.zeros(MAX_SMI_LEN)

    # Create synthetic features for now (you need to load actual ones)
    n_samples = len(all_dataset)
    # Based on your code: 252 one-hot + 148 auxiliary = 400 total
    # Cell line features: columns 254-331 (78 features)
    # Concentration features: columns 252-253
    n_features = 400

    X = np.zeros((n_samples, n_features))
    # Set random values for testing
    X[:, 252] = all_dataset['Conc1'].values if 'Conc1' in all_dataset.columns else np.random.rand(n_samples)
    X[:, 253] = all_dataset['Conc2'].values if 'Conc2' in all_dataset.columns else np.random.rand(n_samples)
    X[:, 254:254+78] = np.random.randn(n_samples, 78)  # Cell line features

    # Use PercentageGrowth as synergy
    y = all_dataset['PercentageGrowth'].values.reshape(-1, 1) if 'PercentageGrowth' in all_dataset.columns else np.random.randn(n_samples, 1)

    return {
        'all_dataset': all_dataset,
        'all_SMILES': all_SMILES,
        'X': X,
        'y': y,
        'smiles_dict': smiles_dict,
        'drug_smiles_cache': drug_smiles_cache,
        'MAX_SMI_LEN': MAX_SMI_LEN,
        'n_cell_features': 78  # From your code: 254:254+78
    }

# ==============================
# 2. FIXED DATA GENERATOR
# ==============================

def prepare_single_combo_batch(data_dict, combo_id, doses_per_combo=5):
    """Prepare a single combination with multiple doses - FIXED VERSION"""
    all_dataset = data_dict['all_dataset']
    drug_smiles_cache = data_dict['drug_smiles_cache']
    X = data_dict['X']
    MAX_SMI_LEN = data_dict['MAX_SMI_LEN']

    # Get all doses for this combination
    combo_data = all_dataset[all_dataset['combo_id'] == combo_id]

    if len(combo_data) < 2:  # Need at least 2 doses for ranking
        return None

    # Take up to doses_per_combo doses
    n_doses = min(len(combo_data), doses_per_combo)

    # Prepare arrays
    drug1_seqs = []
    drug2_seqs = []
    conc1_vals = []
    conc2_vals = []
    cell_features = []
    synergy_vals = []

    # Take the first n_doses samples
    for i in range(n_doses):
        idx = combo_data.index[i]
        row = all_dataset.loc[idx]

        # Get drug SMILES
        drug1_seq = drug_smiles_cache.get(row['Drug1'], np.zeros(MAX_SMI_LEN))
        drug2_seq = drug_smiles_cache.get(row['Drug2'], np.zeros(MAX_SMI_LEN))

        # Get concentrations (from X matrix or dataset)
        conc1 = X[idx, 252] if X.shape[1] > 252 else row.get('Conc1', 0)
        conc2 = X[idx, 253] if X.shape[1] > 253 else row.get('Conc2', 0)

        # Get cell line features
        cell_feat = X[idx, 254:254+78] if X.shape[1] >= 332 else np.zeros(78)

        # Get synergy
        synergy = data_dict['y'][idx][0]

        drug1_seqs.append(drug1_seq)
        drug2_seqs.append(drug2_seq)
        conc1_vals.append(conc1)
        conc2_vals.append(conc2)
        cell_features.append(cell_feat)
        synergy_vals.append(synergy)

    # Convert to numpy arrays with proper shapes
    drug1_array = np.array(drug1_seqs)  # Shape: (n_doses, MAX_SMI_LEN)
    drug2_array = np.array(drug2_seqs)  # Shape: (n_doses, MAX_SMI_LEN)
    conc1_array = np.array(conc1_vals).reshape(-1, 1)  # Shape: (n_doses, 1)
    conc2_array = np.array(conc2_vals).reshape(-1, 1)  # Shape: (n_doses, 1)
    cell_array = np.array(cell_features)  # Shape: (n_doses, 78)
    synergy_array = np.array(synergy_vals).reshape(-1, 1)  # Shape: (n_doses, 1)

    return {
        'drug1': drug1_array,
        'drug2': drug2_array,
        'conc1': conc1_array,
        'conc2': conc2_array,
        'cell_line': cell_array,
        'synergy': synergy_array,
        'n_doses': n_doses
    }


class URankDataGenerator(keras.utils.Sequence):
    """Fixed data generator for uRank training"""
    def __init__(self, data_dict, combo_ids, batch_size=8, doses_per_combo=5, shuffle=True):
        self.data_dict = data_dict
        self.combo_ids = [cid for cid in combo_ids if len(data_dict['all_dataset'][data_dict['all_dataset']['combo_id'] == cid]) >= 2]
        self.batch_size = batch_size
        self.doses_per_combo = doses_per_combo
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.combo_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.combo_ids))
        batch_combo_ids = self.combo_ids[batch_start:batch_end]

        # We'll process each combination separately
        all_drug1 = []
        all_drug2 = []
        all_conc1 = []
        all_conc2 = []
        all_cell = []
        all_synergy = []

        for combo_id in batch_combo_ids:
            combo_data = prepare_single_combo_batch(
                self.data_dict, combo_id, self.doses_per_combo
            )

            if combo_data is not None:
                all_drug1.append(combo_data['drug1'])
                all_drug2.append(combo_data['drug2'])
                all_conc1.append(combo_data['conc1'])
                all_conc2.append(combo_data['conc2'])
                all_cell.append(combo_data['cell_line'])
                all_synergy.append(combo_data['synergy'])

        if not all_drug1:  # If no valid combinations
            # Return empty arrays with correct shapes
            empty_shape = (len(batch_combo_ids), self.doses_per_combo)
            return (
                {
                    'drug1_input': np.zeros(empty_shape + (self.data_dict['MAX_SMI_LEN'],)),
                    'drug2_input': np.zeros(empty_shape + (self.data_dict['MAX_SMI_LEN'],)),
                    'conc1_input': np.zeros(empty_shape + (1,)),
                    'conc2_input': np.zeros(empty_shape + (1,)),
                    'cell_input': np.zeros(empty_shape + (self.data_dict['n_cell_features'],))
                },
                np.zeros(empty_shape + (1,))
            )

        # Stack along batch dimension
        # The shape should be: (batch_size, doses_per_combo, feature_dim)
        drug1_batch = np.stack(all_drug1)  # Shape: (batch_size, doses_per_combo, MAX_SMI_LEN)
        drug2_batch = np.stack(all_drug2)
        conc1_batch = np.stack(all_conc1)  # Shape: (batch_size, doses_per_combo, 1)
        conc2_batch = np.stack(all_conc2)
        cell_batch = np.stack(all_cell)    # Shape: (batch_size, doses_per_combo, 78)
        synergy_batch = np.stack(all_synergy)  # Shape: (batch_size, doses_per_combo, 1)

        # Reshape for the model - we need to process each dose separately
        # Flatten the batch and doses dimensions
        batch_size, n_doses, smi_len = drug1_batch.shape
        drug1_flat = drug1_batch.reshape(-1, smi_len)  # Shape: (batch_size * n_doses, MAX_SMI_LEN)
        drug2_flat = drug2_batch.reshape(-1, smi_len)
        conc1_flat = conc1_batch.reshape(-1, 1)  # Shape: (batch_size * n_doses, 1)
        conc2_flat = conc2_batch.reshape(-1, 1)
        cell_flat = cell_batch.reshape(-1, self.data_dict['n_cell_features'])  # Shape: (batch_size * n_doses, 78)
        synergy_flat = synergy_batch.reshape(-1, 1)  # Shape: (batch_size * n_doses, 1)

        return (
            {
                'drug1_input': drug1_flat,
                'drug2_input': drug2_flat,
                'dose1_input': conc1_flat,
                'dose2_input': conc2_flat,
                'cell_line_input': cell_flat
            },
            synergy_flat
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.combo_ids)


# ==============================
# 3. CUSTOM LOSSES AND LAYERS
# ==============================

class SinusoidalDoseEncoder(layers.Layer):
    """Sinusoidal encoding for dose values"""
    def __init__(self, embedding_dims=64, **kwargs):
        super(SinusoidalDoseEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims

    def call(self, inputs):
        # inputs shape: (batch_size, 1)
        r = tf.expand_dims(inputs, -1)  # Shape: (batch_size, 1, 1)
        i = tf.range(self.embedding_dims, dtype=tf.float32)  # i = 0,1,...,63

        # Calculate 10^(-4i/64)
        exponents = -4.0 * i / self.embedding_dims
        factors = tf.pow(10.0, exponents)  # Shape: (64,)

        # Calculate argument: r * factors
        argument = r * tf.reshape(factors, [1, 1, -1])  # Shape: (batch_size, 1, 64)

        # Apply sin/cos based on even/odd indices
        # Create masks
        sin_mask = tf.cast(i % 2 == 0, tf.float32)  # Shape: (64,)
        cos_mask = tf.cast(i % 2 == 1, tf.float32)  # Shape: (64,)

        sin_component = tf.reshape(sin_mask, [1, 1, -1]) * tf.math.sin(argument)
        cos_component = tf.reshape(cos_mask, [1, 1, -1]) * tf.math.cos(argument)

        encoding = sin_component + cos_component  # Shape: (batch_size, 1, 64)

        # Remove the middle dimension
        encoding = tf.squeeze(encoding, axis=1)  # Shape: (batch_size, 64)

        return encoding

    def get_config(self):
        config = super().get_config()
        config.update({'embedding_dims': self.embedding_dims})
        return config


class URankLoss(keras.losses.Loss):
    """Implementation of uRank loss for single combination"""
    def __init__(self, name='urank_loss', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: synergy values for one combination, shape (n_doses, 1)
        y_pred: predicted synergy values, shape (n_doses, 1)
        """
        # Reshape to (n_doses, 1) if needed
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, -1)
        if len(y_pred.shape) == 1:
            y_pred = tf.expand_dims(y_pred, -1)

        n = tf.shape(y_true)[0]

        # Step 1: G = (2^s - 1)
        G = tf.math.pow(2.0, y_true) - 1.0

        # Step 2: P = Se^T - eS^T
        e = tf.ones_like(y_true)  # Shape: (n_doses, 1)
        S = y_true

        # For matrix multiplication
        S_T = tf.transpose(S)  # Shape: (1, n_doses)
        e_T = tf.transpose(e)  # Shape: (1, n_doses)

        # P = S @ e^T - e @ S^T
        term1 = tf.matmul(S, e_T)  # Shape: (n_doses, n_doses)
        term2 = tf.matmul(e, S_T)  # Shape: (n_doses, n_doses)
        P = term1 - term2

        # Step 3: M = 1 if P > 0, else 0
        M = tf.cast(tf.greater(P, 0), tf.float32)

        # Step 4: T = M * exp(O)
        O = y_pred
        T = M * tf.math.exp(O)

        # Step 5: L = ln(1 + T / exp(O))
        L = tf.math.log(1.0 + T / tf.math.exp(O))

        # Step 6: L = L ⊙ G (element-wise multiplication)
        L = L * G

        # Step 7: loss = sum(L) / (n-1)
        # We need to average over all doses
        loss = tf.reduce_sum(L) / tf.cast(n - 1, tf.float32)

        return loss


class HybridLoss(keras.losses.Loss):
    """Combined MSE + uRank loss"""
    def __init__(self, lambda_param=0.5, name='hybrid_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_param = lambda_param
        self.mse = keras.losses.MeanSquaredError()
        self.urank = URankLoss()

    def call(self, y_true, y_pred):
        # For uRank, we need to process each combination separately
        # We'll assume y_true and y_pred contain doses for one combination

        # Calculate MSE
        mse_loss = self.mse(y_true, y_pred)

        # Calculate uRank loss
        urank_loss = self.urank(y_true, y_pred)

        # Combine
        return self.lambda_param * mse_loss + (1 - self.lambda_param) * urank_loss


# ==============================
# 4. CDCR-RANK MODEL ARCHITECTURE
# ==============================

def build_cdcr_rank_model(MAX_SMI_LEN=100, n_cell_features=78,
                         drug_embed_dim=128, dose_embed_dim=64):
    """Build the complete CDCR-Rank model"""

    # ===== Input Layers =====
    # Drug inputs (SMILES sequences) - Shape: (batch_size, MAX_SMI_LEN)
    drug1_input = layers.Input(shape=(MAX_SMI_LEN,), name='drug1_input', dtype='float32')
    drug2_input = layers.Input(shape=(MAX_SMI_LEN,), name='drug2_input', dtype='float32')

    # Dose inputs - Shape: (batch_size, 1)
    dose1_input = layers.Input(shape=(1,), name='dose1_input')
    dose2_input = layers.Input(shape=(1,), name='dose2_input')

    # Cell line input - Shape: (batch_size, n_cell_features)
    cell_line_input = layers.Input(shape=(n_cell_features,), name='cell_line_input')

    # ===== Drug Encoder (1D-CNN) =====
    # Since drug inputs are already encoded as integers, we need an Embedding layer
    # First, convert float to int (since SMILES are encoded as integers)
    drug1_int = layers.Lambda(lambda x: tf.cast(x, tf.int32))(drug1_input)
    drug2_int = layers.Lambda(lambda x: tf.cast(x, tf.int32))(drug2_input)

    # Embedding layer for SMILES
    smiles_embedding = layers.Embedding(
        input_dim=65,  # 64 characters + 1 for padding
        output_dim=drug_embed_dim,
        input_length=MAX_SMI_LEN,
        mask_zero=True,
        name='smiles_embedding'
    )

    # 1D CNN layers
    def build_drug_cnn():
        return keras.Sequential([
            layers.Conv1D(32, 4, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 4, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 4, activation='relu', padding='same'),
            layers.GlobalMaxPooling1D()
        ], name='drug_cnn')

    drug_cnn = build_drug_cnn()

    # Process both drugs
    drug1_embedded = smiles_embedding(drug1_int)  # Shape: (batch_size, MAX_SMI_LEN, drug_embed_dim)
    drug2_embedded = smiles_embedding(drug2_int)

    drug1_encoded = drug_cnn(drug1_embedded)  # Shape: (batch_size, 128)
    drug2_encoded = drug_cnn(drug2_embedded)  # Shape: (batch_size, 128)

    # ===== Dose Encoder (Sinusoidal) =====
    dose_encoder = SinusoidalDoseEncoder(embedding_dims=dose_embed_dim)
    dose1_encoded = dose_encoder(dose1_input)  # Shape: (batch_size, 64)
    dose2_encoded = dose_encoder(dose2_input)  # Shape: (batch_size, 64)

    # ===== Cell Line Encoder (MLP) =====
    cell_encoded = layers.Dense(128, activation='relu')(cell_line_input)
    cell_encoded = layers.Dense(256, activation='relu')(cell_encoded)
    cell_encoded = layers.Dense(128, activation='relu')(cell_encoded)  # Shape: (batch_size, 128)

    # ===== Concatenate Features =====
    concatenated = layers.Concatenate()([
        drug1_encoded, drug2_encoded,
        dose1_encoded, dose2_encoded,
        cell_encoded
    ])  # Shape: (batch_size, 128+128+64+64+128 = 512)

    # ===== Prediction Network =====
    x = layers.Dense(256, activation='relu')(concatenated)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output layer
    synergy_output = layers.Dense(1, name='synergy_output')(x)

    # ===== Create Model =====
    model = Model(
        inputs=[drug1_input, drug2_input, dose1_input, dose2_input, cell_line_input],
        outputs=synergy_output,
        name='CDCR_Rank'
    )

    return model


# ==============================
# 5. TRAINING FUNCTION
# ==============================

def train_cdcr_rank(data_dict, train_combo_ids, val_combo_ids,
                   epochs=30, batch_size=4, doses_per_combo=5):
    """Train CDCR-Rank model"""

    print(f"Training with {len(train_combo_ids)} combinations, validating with {len(val_combo_ids)}")

    # Create data generators
    train_gen = URankDataGenerator(
        data_dict, train_combo_ids,
        batch_size=batch_size,
        doses_per_combo=doses_per_combo,
        shuffle=True
    )

    val_gen = URankDataGenerator(
        data_dict, val_combo_ids,
        batch_size=batch_size,
        doses_per_combo=doses_per_combo,
        shuffle=False
    )

    # Build model
    model = build_cdcr_rank_model(
        MAX_SMI_LEN=data_dict['MAX_SMI_LEN'],
        n_cell_features=data_dict['n_cell_features']
    )

    print("\nModel Summary:")
    model.summary()

    # Compile with hybrid loss
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = HybridLoss(lambda_param=0.5)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.MeanSquaredError(name='mse'),
            keras.metrics.MeanAbsoluteError(name='mae')
        ]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    # Train
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# ==============================
# 6. MAIN EXECUTION
# ==============================

def main():
    print("=" * 60)
    print("CDCR-RANK: Drug Combination Dose Response Prediction")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    data_path = '/data/'  # UPDATE THIS PATH

    if True:
        data_dict = load_nci_almanac_data(data_path)
        print(f"✓ Loaded {len(data_dict['all_dataset'])} samples")
        print(f"✓ {len(data_dict['drug_smiles_cache'])} drugs in cache")
    """except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Please check your data path and file formats")
        return"""

    # Get unique combinations
    unique_combos = data_dict['all_dataset']['combo_id'].unique()
    print(f"\n2. Found {len(unique_combos)} unique drug-cell line combinations")

    # Filter combinations with at least 2 doses
    valid_combos = []
    for combo in unique_combos:
        n_doses = len(data_dict['all_dataset'][data_dict['all_dataset']['combo_id'] == combo])
        if n_doses >= 2:
            valid_combos.append(combo)

    print(f"✓ {len(valid_combos)} combinations have ≥2 doses (required for uRank)")

    # Split
    np.random.seed(42)
    np.random.shuffle(valid_combos)

    train_size = int(0.7 * len(valid_combos))
    val_size = int(0.15 * len(valid_combos))

    train_combos = valid_combos[:train_size]
    val_combos = valid_combos[train_size:train_size + val_size]
    test_combos = valid_combos[train_size + val_size:]

    print(f"\n3. Data split:")
    print(f"   Training: {len(train_combos)} combinations")
    print(f"   Validation: {len(val_combos)} combinations")
    print(f"   Test: {len(test_combos)} combinations")

    # Train model
    print("\n4. Building and training model...")
    model, history = train_cdcr_rank(
        data_dict=data_dict,
        train_combo_ids=train_combos,
        val_combo_ids=val_combos,
        epochs=20,  # Reduced for testing
        batch_size=4,
        doses_per_combo=3  # Use 3 doses per combination
    )

    # Save model
    print("\n5. Saving model...")
    model.save('cdcr_rank_model.h5')
    print("✓ Model saved as 'cdcr_rank_model.h5'")

    # Plot training history
    print("\n6. Training complete!")
    print("\nFinal metrics:")
    if hasattr(history, 'history'):
        print(f"   Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"   Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"   Training MSE: {history.history['mse'][-1]:.4f}")
        print(f"   Validation MSE: {history.history['val_mse'][-1]:.4f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    main()

# ==============================
# 7. INTEGRATED GRADIENTS FOR ATTRIBUTION
# ==============================

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class IntegratedGradients:
    """Compute Integrated Gradients attribution"""
    def __init__(self, model, baseline=None):
        self.model = model
        self.baseline = baseline

    def compute_gradients(self, inputs, target_class_idx=0):
        """Compute gradients with respect to inputs"""
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.model(inputs, training=False)
            loss = predictions[:, target_class_idx]
        return tape.gradient(loss, inputs)

    def integrated_gradients(self, inputs, steps=50):
        """Compute integrated gradients"""
        if self.baseline is None:
            # Use zero baseline for embeddings
            self.baseline = [tf.zeros_like(inp) for inp in inputs]

        # Initialize attributions
        attributions = [tf.zeros_like(inp) for inp in inputs]

        # Compute integrated gradients
        for alpha in np.linspace(0, 1, steps):
            # Interpolated inputs
            interpolated_inputs = [
                baseline + alpha * (input_val - baseline)
                for baseline, input_val in zip(self.baseline, inputs)
            ]

            # Compute gradients
            gradients = self.compute_gradients(interpolated_inputs)

            # Add to attributions
            for i in range(len(attributions)):
                attributions[i] += gradients[i] * (1.0 / steps)

        # Multiply by (inputs - baseline)
        final_attributions = []
        for i in range(len(attributions)):
            final_attributions.append(
                attributions[i] * (inputs[i] - self.baseline[i])
            )

        return final_attributions

# ==============================
# 8. ATTRIBUTION-GUIDED ABLATION EXPERIMENT
# ==============================

def identify_top_synergy_pairs(model, test_data_dict, threshold=0.8, top_k=50):
    """
    Identify top predicted synergistic pairs
    Returns list of (combo_id, max_synergy_score, sample_index)
    """
    print("\n" + "="*60)
    print("IDENTIFYING TOP SYNERGY PREDICTIONS")
    print("="*60)

    top_pairs = []
    all_dataset = test_data_dict['all_dataset']

    # Get unique combinations
    unique_combos = all_dataset['combo_id'].unique()
    print(f"Evaluating {len(unique_combos)} unique combinations...")

    for combo_id in tqdm(unique_combos[:1000]):  # Limit for speed
        combo_data = all_dataset[all_dataset['combo_id'] == combo_id]

        if len(combo_data) < 2:
            continue

        # Take first sample for prediction
        sample_idx = combo_data.index[0]
        row = all_dataset.loc[sample_idx]

        # Prepare input
        drug1_seq = test_data_dict['drug_smiles_cache'].get(
            row['Drug1'], np.zeros(test_data_dict['MAX_SMI_LEN'])
        )
        drug2_seq = test_data_dict['drug_smiles_cache'].get(
            row['Drug2'], np.zeros(test_data_dict['MAX_SMI_LEN'])
        )

        conc1 = test_data_dict['X'][sample_idx, 252] if test_data_dict['X'].shape[1] > 252 else 0.1
        conc2 = test_data_dict['X'][sample_idx, 253] if test_data_dict['X'].shape[1] > 253 else 0.1
        cell_feat = test_data_dict['X'][sample_idx, 254:254+78] if test_data_dict['X'].shape[1] >= 332 else np.zeros(78)

        # Make prediction
        inputs = {
            'drug1_input': np.array([drug1_seq]),
            'drug2_input': np.array([drug2_seq]),
            'dose1_input': np.array([[conc1]]),
            'dose2_input': np.array([[conc2]]),
            'cell_line_input': np.array([cell_feat])
        }

        prediction = model.predict(inputs, verbose=0)[0][0]

        if prediction > threshold:
            top_pairs.append((combo_id, prediction, sample_idx))

    # Sort and take top K
    top_pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = top_pairs[:top_k]

    print(f"\nFound {len(top_pairs)} pairs with synergy > {threshold}")
    print(f"Top 5 pairs:")
    for i, (combo_id, score, idx) in enumerate(top_pairs[:5]):
        print(f"  {i+1}. {combo_id}: score = {score:.3f}")

    return top_pairs

def compute_attributions_for_sample(model, sample_idx, data_dict):
    """Compute Integrated Gradients attributions for a sample"""
    # Get sample data
    row = data_dict['all_dataset'].loc[sample_idx]

    drug1_seq = data_dict['drug_smiles_cache'].get(
        row['Drug1'], np.zeros(data_dict['MAX_SMI_LEN'])
    )
    drug2_seq = data_dict['drug_smiles_cache'].get(
        row['Drug2'], np.zeros(data_dict['MAX_SMI_LEN'])
    )

    conc1 = data_dict['X'][sample_idx, 252] if data_dict['X'].shape[1] > 252 else 0.1
    conc2 = data_dict['X'][sample_idx, 253] if data_dict['X'].shape[1] > 253 else 0.1
    cell_feat = data_dict['X'][sample_idx, 254:254+78] if data_dict['X'].shape[1] >= 332 else np.zeros(78)

    # Prepare inputs
    drug1_input = tf.constant([drug1_seq], dtype=tf.float32)
    drug2_input = tf.constant([drug2_seq], dtype=tf.float32)
    dose1_input = tf.constant([[conc1]], dtype=tf.float32)
    dose2_input = tf.constant([[conc2]], dtype=tf.float32)
    cell_input = tf.constant([cell_feat], dtype=tf.float32)

    inputs = [drug1_input, drug2_input, dose1_input, dose2_input, cell_input]

    # Compute Integrated Gradients
    ig = IntegratedGradients(model)
    attributions = ig.integrated_gradients(inputs, steps=20)

    # Convert to numpy and get magnitudes
    attr_dict = {
        'drug1': np.abs(attributions[0].numpy()[0]),  # Shape: (100,)
        'drug2': np.abs(attributions[1].numpy()[0]),  # Shape: (100,)
        'dose1': np.abs(attributions[2].numpy()[0]),  # Shape: (1,)
        'dose2': np.abs(attributions[3].numpy()[0]),  # Shape: (1,)
        'cell': np.abs(attributions[4].numpy()[0])    # Shape: (78,)
    }

    return attr_dict, row, (drug1_seq, drug2_seq, conc1, conc2, cell_feat)

def attribution_guided_ablation(model, top_pairs, data_dict, attribution_threshold=0.6):
    """
    Perform attribution-guided ablation experiment
    """
    print("\n" + "="*60)
    print("ATTRIBUTION-GUIDED ABLATION EXPERIMENT")
    print("="*60)

    results = []

    for combo_id, original_score, sample_idx in tqdm(top_pairs):
        # 1. Compute attributions
        attr_dict, row, original_inputs = compute_attributions_for_sample(model, sample_idx, data_dict)

        # 2. Identify critical features for each modality
        critical_features = {}

        for modality in ['drug1', 'drug2', 'cell']:
            attributions = attr_dict[modality]
            max_attr = np.max(attributions)
            threshold = attribution_threshold * max_attr

            # Find dimensions where attribution > threshold
            critical_indices = np.where(attributions > threshold)[0]
            critical_features[modality] = critical_indices

        # 3. Perform ablations and record prediction changes
        ablation_results = {}

        # Original prediction
        drug1_seq, drug2_seq, conc1, conc2, cell_feat = original_inputs

        # Test Drug1 ablation
        if len(critical_features['drug1']) > 0:
            drug1_ablated = drug1_seq.copy()
            drug1_ablated[critical_features['drug1']] = 0

            inputs = {
                'drug1_input': np.array([drug1_ablated]),
                'drug2_input': np.array([drug2_seq]),
                'dose1_input': np.array([[conc1]]),
                'dose2_input': np.array([[conc2]]),
                'cell_line_input': np.array([cell_feat])
            }
            pred_drug1_ablated = model.predict(inputs, verbose=0)[0][0]
            ablation_results['drug1'] = original_score - pred_drug1_ablated
        else:
            ablation_results['drug1'] = 0.0

        # Test Drug2 ablation
        if len(critical_features['drug2']) > 0:
            drug2_ablated = drug2_seq.copy()
            drug2_ablated[critical_features['drug2']] = 0

            inputs = {
                'drug1_input': np.array([drug1_seq]),
                'drug2_input': np.array([drug2_ablated]),
                'dose1_input': np.array([[conc1]]),
                'dose2_input': np.array([[conc2]]),
                'cell_line_input': np.array([cell_feat])
            }
            pred_drug2_ablated = model.predict(inputs, verbose=0)[0][0]
            ablation_results['drug2'] = original_score - pred_drug2_ablated
        else:
            ablation_results['drug2'] = 0.0

        # Test Cell ablation
        if len(critical_features['cell']) > 0:
            cell_ablated = cell_feat.copy()
            cell_ablated[critical_features['cell']] = 0

            inputs = {
                'drug1_input': np.array([drug1_seq]),
                'drug2_input': np.array([drug2_seq]),
                'dose1_input': np.array([[conc1]]),
                'dose2_input': np.array([[conc2]]),
                'cell_line_input': np.array([cell_ablated])
            }
            pred_cell_ablated = model.predict(inputs, verbose=0)[0][0]
            ablation_results['cell'] = original_score - pred_cell_ablated
        else:
            ablation_results['cell'] = 0.0

        # 4. Determine dominance
        total_change = sum(ablation_results.values())
        if total_change > 0:
            relative_importance = {
                modality: change / total_change
                for modality, change in ablation_results.items()
            }

            # Find dominant modality
            dominant_modality = max(relative_importance.items(), key=lambda x: x[1])[0]

            dominance_type = 'balanced'
            if relative_importance[dominant_modality] > 0.5:
                dominance_type = f'{dominant_modality}-dominant'

        # Store results
        results.append({
            'combo_id': combo_id,
            'drug1': row['Drug1'],
            'drug2': row['Drug2'],
            'cell_line': row['CellLine'],
            'original_score': original_score,
            'ablation_changes': ablation_results,
            'critical_features': critical_features,
            'dominance_type': dominance_type,
            'sample_idx': sample_idx
        })

    return results

def analyze_and_visualize_results(results):
    """
    Analyze and visualize ablation experiment results
    """
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("="*60)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # 1. Summary statistics
    print("\n1. SUMMARY STATISTICS:")
    print(f"Total samples analyzed: {len(df_results)}")

    # Dominance distribution
    dominance_counts = df_results['dominance_type'].value_counts()
    print("\nDominance distribution:")
    for dom_type, count in dominance_counts.items():
        percentage = count / len(df_results) * 100
        print(f"  {dom_type}: {count} samples ({percentage:.1f}%)")

    # 2. Mean ablation effects
    print("\n2. MEAN ABLATION EFFECTS:")
    ablation_means = {}
    for modality in ['drug1', 'drug2', 'cell']:
        changes = [r['ablation_changes'][modality] for r in results]
        ablation_means[modality] = np.mean(changes)
        print(f"  {modality}: Δ = {ablation_means[modality]:.4f}")

    # 3. Case studies (top 3 by original score)
    print("\n3. CASE STUDIES (Top 3 predictions):")
    df_sorted = df_results.sort_values('original_score', ascending=False)

    for i in range(min(3, len(df_sorted))):
        case = df_sorted.iloc[i]
        print(f"\n  Case {i+1}: {case['drug1']} + {case['drug2']} in {case['cell_line']}")
        print(f"    Original score: {case['original_score']:.3f}")
        print(f"    Dominance type: {case['dominance_type']}")
        print(f"    Ablation changes:")
        for modality, change in case['ablation_changes'].items():
            print(f"      {modality}: Δ = {change:.3f}")
        print(f"    Critical features found:")
        for modality, indices in case['critical_features'].items():
            print(f"      {modality}: {len(indices)} dimensions")

    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Dominance distribution
    axes[0, 0].pie(dominance_counts.values, labels=dominance_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Modality Dominance Distribution')

    # Plot 2: Ablation effects
    modalities = list(ablation_means.keys())
    means = list(ablation_means.values())
    axes[0, 1].bar(modalities, means, color=['red', 'blue', 'green'])
    axes[0, 1].set_ylabel('Mean Prediction Change (Δ)')
    axes[0, 1].set_title('Mean Ablation Effects')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Original vs ablated scores for top cases
    top_cases = df_sorted.head(5)
    x_pos = np.arange(len(top_cases))
    width = 0.25

    for i, modality in enumerate(['drug1', 'drug2', 'cell']):
        changes = [case['ablation_changes'][modality] for _, case in top_cases.iterrows()]
        axes[1, 0].bar(x_pos + i*width, changes, width, label=modality)

    axes[1, 0].set_xlabel('Top Cases')
    axes[1, 0].set_ylabel('Prediction Change (Δ)')
    axes[1, 0].set_title('Ablation Effects for Top 5 Cases')
    axes[1, 0].set_xticks(x_pos + width)
    axes[1, 0].set_xticklabels([f"Case {i+1}" for i in range(len(top_cases))])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Critical features distribution
    avg_critical_features = {}
    for modality in ['drug1', 'drug2', 'cell']:
        avg_critical_features[modality] = np.mean([
            len(r['critical_features'][modality]) for r in results
        ])

    axes[1, 1].bar(avg_critical_features.keys(), avg_critical_features.values(),
                   color=['red', 'blue', 'green'])
    axes[1, 1].set_ylabel('Average Critical Features')
    axes[1, 1].set_title('Average Critical Features per Modality')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('interpretability_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    return df_results

# ==============================
# 9. RUN INTERPRETABILITY EXPERIMENT
# ==============================

def run_interpretability_experiment(model_path=None, data_path=None, threshold=0.8, top_k=50):
    """
    Main function to run the complete interpretability experiment
    """
    print("\n" + "="*60)
    print("DRUG SYNERGY INTERPRETABILITY EXPERIMENT")
    print("="*60)

    # Load or train model
    if model_path and os.path.exists(model_path):
        print(f"\n1. Loading pre-trained model from {model_path}")
        model = keras.models.load_model(model_path)
    else:
        print("\n1. Training new model (this may take a while)...")
        # Use your existing training code
        from your_main_code import main
        model, _ = main()  # This should return the trained model

    # Load data
    print("\n2. Loading data...")
    if data_path is None:
        data_path = '/data/'

    data_dict = load_nci_almanac_data(data_path)

    # Step 1: Identify top synergy pairs
    print("\n3. Identifying top synergy predictions...")
    top_pairs = identify_top_synergy_pairs(
        model, data_dict, threshold=threshold, top_k=top_k
    )

    if len(top_pairs) == 0:
        print("No high-confidence synergy pairs found. Try lowering the threshold.")
        return None

    # Step 2: Attribution-guided ablation
    print("\n4. Performing attribution-guided ablation...")
    results = attribution_guided_ablation(model, top_pairs, data_dict)

    # Step 3: Analyze and visualize
    print("\n5. Analyzing results...")
    df_results = analyze_and_visualize_results(results)

    # Save results
    results_df = pd.DataFrame([
        {
            'combo_id': r['combo_id'],
            'drug1': r['drug1'],
            'drug2': r['drug2'],
            'cell_line': r['cell_line'],
            'original_score': r['original_score'],
            'drug1_ablation': r['ablation_changes']['drug1'],
            'drug2_ablation': r['ablation_changes']['drug2'],
            'cell_ablation': r['ablation_changes']['cell'],
            'dominance_type': r['dominance_type'],
            'n_critical_drug1': len(r['critical_features']['drug1']),
            'n_critical_drug2': len(r['critical_features']['drug2']),
            'n_critical_cell': len(r['critical_features']['cell'])
        }
        for r in results
    ])

    results_df.to_csv('interpretability_results.csv', index=False)
    print("\n✓ Results saved to 'interpretability_results.csv'")

    # Print biological interpretation insights
    print("\n" + "="*60)
    print("BIOLOGICAL INTERPRETATION INSIGHTS")
    print("="*60)

    # Analyze by dominance type
    for dom_type in results_df['dominance_type'].unique():
        dom_samples = results_df[results_df['dominance_type'] == dom_type]

        print(f"\n{len(dom_samples)} samples are {dom_type}:")

        if 'drug1-dominant' in dom_type:
            print("  - Synergy primarily driven by Drug 1's chemical features")
            print("  - Suggests Drug 1's mechanism-of-action is key facilitator")
            # Show example
            example = dom_samples.iloc[0]
            print(f"  Example: {example['drug1']} + {example['drug2']} "
                  f"(Δ_drug1 = {example['drug1_ablation']:.3f})")

        elif 'drug2-dominant' in dom_type:
            print("  - Synergy primarily driven by Drug 2's chemical features")
            print("  - Suggests complementary action from Drug 2")
            example = dom_samples.iloc[0]
            print(f"  Example: {example['drug1']} + {example['drug2']} "
                  f"(Δ_drug2 = {example['drug2_ablation']:.3f})")

        elif 'cell-dominant' in dom_type:
            print("  - Synergy highly dependent on cellular context")
            print("  - Suggests pathway vulnerabilities specific to cell line")
            example = dom_samples.iloc[0]
            print(f"  Example: {example['drug1']} + {example['drug2']} in "
                  f"{example['cell_line']} (Δ_cell = {example['cell_ablation']:.3f})")

        elif 'balanced' in dom_type:
            print("  - Synergy requires contributions from multiple modalities")
            print("  - Suggests complex, multi-factorial interaction")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)

    return df_results

# ==============================
# 10. MAIN EXECUTION WITH INTERPRETABILITY
# ==============================

def main_with_interpretability():
    """
    Main function that includes interpretability analysis
    """
    print("CDCR-RANK with Interpretability Analysis")
    print("="*50)

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("\n1. Loading data...")
    data_path = '/data/'
    data_dict = load_nci_almanac_data(data_path)

    # Get unique combos
    unique_combos = data_dict['all_dataset']['combo_id'].unique()
    valid_combos = [c for c in unique_combos
                    if len(data_dict['all_dataset'][data_dict['all_dataset']['combo_id'] == c]) >= 2]

    # Split
    np.random.shuffle(valid_combos)
    train_size = int(0.7 * len(valid_combos))
    val_size = int(0.15 * len(valid_combos))

    train_combos = valid_combos[:train_size]
    val_combos = valid_combos[train_size:train_size + val_size]
    test_combos = valid_combos[train_size + val_size:]

    print(f"\n2. Data split:")
    print(f"   Training: {len(train_combos)} combinations")
    print(f"   Validation: {len(val_combos)} combinations")
    print(f"   Test: {len(test_combos)} combinations")

    # Train model
    print("\n3. Training model...")
    model, history = train_cdcr_rank(
        data_dict=data_dict,
        train_combo_ids=train_combos,
        val_combo_ids=val_combos,
        epochs=20,
        batch_size=4,
        doses_per_combo=3
    )

    # Save model
    model.save('cdcr_rank_model.h5')
    print("✓ Model saved")

    # Run interpretability experiment
    print("\n4. Running interpretability experiment...")
    results = run_interpretability_experiment(
        model_path='cdcr_rank_model.h5',
        data_path=data_path,
        threshold=0.7,  # Lower threshold for demonstration
        top_k=30       # Analyze top 30 pairs
    )

    return model, history, results

# ==============================
# 11. BONUS: FEATURE IMPORTANCE ANALYSIS
# ==============================

def analyze_feature_importance(results, data_dict):
    """
    Analyze which specific features are most important across all samples
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Aggregate critical features across all samples
    drug1_features = []
    drug2_features = []
    cell_features = []

    for r in results:
        drug1_features.extend(r['critical_features']['drug1'])
        drug2_features.extend(r['critical_features']['drug2'])
        cell_features.extend(r['critical_features']['cell'])

    # Count frequencies
    from collections import Counter

    drug1_counter = Counter(drug1_features)
    drug2_counter = Counter(drug2_features)
    cell_counter = Counter(cell_features)

    # Get top features
    top_drug1 = drug1_counter.most_common(10)
    top_drug2 = drug2_counter.most_common(10)
    top_cell = cell_counter.most_common(10)

    print("\nTop important SMILES positions (Drug 1):")
    for pos, count in top_drug1:
        print(f"  Position {pos}: {count} samples")

    print("\nTop important SMILES positions (Drug 2):")
    for pos, count in top_drug2:
        print(f"  Position {pos}: {count} samples")

    print("\nTop important cell line features:")
    for feat_idx, count in top_cell:
        print(f"  Feature {feat_idx}: {count} samples")

    # Map SMILES positions to characters (if possible)
    try:
        smiles_dict = data_dict['smiles_dict']
        reverse_dict = {v: k for k, v in smiles_dict.items()}

        print("\nSMILES character interpretation:")
        for drug_name in ['drug1', 'drug2']:
            for pos, _ in top_drug1[:5]:
                # Get most common character at this position
                chars_at_pos = []
                for r in results:
                    if pos < len(data_dict['drug_smiles_cache'][r[drug_name]]):
                        char_code = int(data_dict['drug_smiles_cache'][r[drug_name]][pos])
                        if char_code in reverse_dict:
                            chars_at_pos.append(reverse_dict[char_code])

                if chars_at_pos:
                    char_counter = Counter(chars_at_pos)
                    most_common_char = char_counter.most_common(1)[0]
                    print(f"  {drug_name} position {pos}: character '{most_common_char[0]}' "
                          f"({most_common_char[1]} samples)")
    except:
        print("\nNote: Could not map SMILES positions to characters")

    return {
        'drug1_features': drug1_counter,
        'drug2_features': drug2_counter,
        'cell_features': cell_counter
    }

# ==============================
# EXECUTE
# ==============================

if __name__ == "__main__":
    import os

    # Uncomment one of these:

    # Option 1: Run quick test
    # quick_test()

    # Option 2: Run full experiment (requires data)
    # main_with_interpretability()

    # Option 3: Run just the interpretability on existing model
    # results = run_interpretability_experiment(
    #     model_path='cdcr_rank_model.h5',
    #     threshold=0.7,
    #     top_k=30
    # )

    print("\nTo run the interpretability experiment:")
    print("1. Ensure you have trained model (cdcr_rank_model.h5)")
    print("2. Uncomment the appropriate function call above")
    print("3. Adjust parameters as needed")

