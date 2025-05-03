import sys
import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_DISABLE_MKL'] = '1'
try:
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import json
    import base64
    from PIL import Image
    import io
    import glob
except ImportError as e:
    print(f"Error: Required library missing: {str(e)}")
    print("Install dependencies: pip install tensorflow numpy pandas pillow")
    sys.exit(1)

SIGN_MEANINGS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to left",
    20: "Dangerous curve to right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5t"
}

def load_data():
    try:
        train_dir = "GTSRB/Training"
        if not os.path.exists(train_dir):
            print("Error: GTSRB dataset not found at 'GTSRB/Training'. Download from http://benchmark.ini.rub.de/")
            sys.exit(1)

        x_train, y_train = [], []
        x_test, y_test = [], []

        for class_id in range(43):
            class_dir = os.path.join(train_dir, f"{class_id:05d}")
            csv_path = os.path.join(class_dir, f"GT-{class_id:05d}.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file for class {class_id} not found at {csv_path}. Skipping class.")
                continue

            annotations = pd.read_csv(csv_path, sep=';')
            image_paths = glob.glob(os.path.join(class_dir, "*.ppm"))
            if len(image_paths) < 10:
                print(f"Warning: Insufficient images for class {class_id} ({len(image_paths)} found). Skipping class.")
                continue

            split_idx = int(0.8 * len(image_paths))
            train_paths = image_paths[:split_idx]
            test_paths = image_paths[split_idx:]

            for img_path in train_paths:
                img = Image.open(img_path).resize((32, 32))
                img_array = np.array(img) / 255.0
                x_train.append(img_array)
                y_train.append(class_id)

            for img_path in test_paths:
                img = Image.open(img_path).resize((32, 32))
                img_array = np.array(img) / 255.0
                x_test.append(img_array)
                y_test.append(class_id)

        if not x_train:
            print("Error: No valid training data loaded. Check dataset integrity.")
            sys.exit(1)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        print(f"Successfully loaded {len(x_train)} training samples and {len(x_test)} test samples across 43 classes.")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"Failed to load GTSRB dataset: {str(e)}. Ensure the 'GTSRB/Training' directory exists.")
        sys.exit(1)

def image_to_base64(img_array):
    try:
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        sys.exit(1)

def train_model():
    try:
        (x_train, y_train), (x_test, y_test) = load_data()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(43, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        log_file = 'training_log.json'
        log_data = []
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                log_entry = {
                    'epoch': epoch + 1,
                    'loss': logs.get('loss'),
                    'accuracy': logs.get('accuracy')
                }
                log_data.append(log_entry)
                try:
                    with open(log_file, 'w') as f:
                        json.dump(log_data, f, indent=2)
                except Exception as e:
                    print(f"Error saving training log to {log_file}: {str(e)}")
        model.fit(x_train, y_train, epochs=10, callbacks=[ProgressCallback()])
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {accuracy:.4f}")

        predictions = []
        indices = np.random.choice(len(x_test), 5, replace=False)
        pred_probs = model.predict(x_test[indices])
        pred_labels = np.argmax(pred_probs, axis=1)
        for i, idx in enumerate(indices):
            predictions.append({
                'image': image_to_base64(x_test[idx]),
                'true_label': int(y_test[idx]),
                'pred_label': int(pred_labels[i])
            })
        with open('predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)

        model.save('model.h5')
        return f"Training completed successfully. Test accuracy: {accuracy:.4f}"
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)

def predict_image(image_path):
    try:
        if not os.path.exists('model.h5'):
            print("Error: Trained model file 'model.h5' not found. Run training first.")
            sys.exit(1)
        model = tf.keras.models.load_model('model.h5')
        img = Image.open(image_path)
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred_probs = model.predict(img_array)
        pred_label = np.argmax(pred_probs, axis=1)[0]
        confidence = np.max(pred_probs)
        meaning = SIGN_MEANINGS.get(pred_label, "Unknown")
        result = {
            'image': image_to_base64(img_array[0]),
            'meaning': meaning,
            'confidence': float(confidence)
        }
        with open('prediction_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Model")
    parser.add_argument('--train', action='store_true', help='Train the model on GTSRB dataset')
    parser.add_argument('--predict', help='Path to an image for prediction')
    args = parser.parse_args()

    try:
        if args.train:
            result = train_model()
            print(result)
        elif args.predict:
            result = predict_image(args.predict)
            print(json.dumps(result, indent=2))
        else:
            print("Specify either --train or --predict <image_path>")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)