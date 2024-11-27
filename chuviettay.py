import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import io


class DigitRecognizer:
    def __init__(self):
        # Thiết lập seed cho tính ổn định
        tf.random.set_seed(42)
        np.random.seed(42)

        # Load và xử lý dữ liệu MNIST
        self.load_and_prepare_data()

        # Xây dựng và compile mô hình
        self.model = self.build_model()

        # Khởi tạo data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )

    def load_and_prepare_data(self):
        """Load và tiền xử lý dữ liệu MNIST"""
        # Load dữ liệu
        (self.X_train, self.y_train), (self.X_test,
                                       self.y_test) = tf.keras.datasets.mnist.load_data()

        # Reshape và normalize
        self.X_train = self.X_train.reshape(-1,
                                            28, 28, 1).astype('float32') / 255.0
        self.X_test = self.X_test.reshape(-1,
                                          28, 28, 1).astype('float32') / 255.0

        # One-hot encoding
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    def build_model(self):
        """Xây dựng mô hình CNN cải tiến"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), padding='same',
                   activation='relu', input_shape=(28, 28, 1)),
            BatchNormalization(),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second Convolutional Block
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        # Compile với Adam optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, epochs=10, batch_size=128):
        """Train mô hình với data augmentation"""
        # Callback để lưu model tốt nhất
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Early stopping để tránh overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )

        # Training với data augmentation
        history = self.model.fit(
            self.datagen.flow(self.X_train, self.y_train,
                              batch_size=batch_size),
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
            callbacks=[checkpoint, early_stopping]
        )

        return history

    def preprocess_image(self, image):
        """Xử lý ảnh đầu vào"""
        # Chuyển về ảnh xám
        if len(np.array(image).shape) == 3:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            image = np.array(image)

        # Đảo ngược màu nếu cần (để phù hợp với MNIST)
        if np.mean(image) > 127:
            image = 255 - image

        # Cân chỉnh độ tương phản
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Loại bỏ nhiễu
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Tìm và cắt vùng chứa chữ số
        thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Lấy contour lớn nhất
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            image = image[y:y+h, x:x+w]

        # Resize về 20x20 và thêm padding để được 28x28
        image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)
        image = np.pad(image, ((4, 4), (4, 4)), 'constant', constant_values=0)

        # Normalize và reshape
        image = image.reshape(1, 28, 28, 1).astype('float32') / 255.0

        return image

    def predict(self, image):
        """Nhận dạng chữ số từ ảnh"""
        # Tiền xử lý ảnh
        processed_image = self.preprocess_image(image)

        # Dự đoán
        predictions = self.model.predict(processed_image)
        digit = np.argmax(predictions[0])
        confidence = float(predictions[0][digit])

        return digit, confidence, predictions[0]


def plot_predictions(predictions):
    """Vẽ biểu đồ confidence scores"""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(10)
    ax.bar(x, predictions)
    ax.set_xticks(x)
    ax.set_xlabel('Chữ số')
    ax.set_ylabel('Độ tin cậy')
    ax.set_title('Confidence Scores')
    return fig


def main():
    st.title("Nhận Dạng Chữ Số Viết Tay")

    # Khởi tạo model
    if 'model' not in st.session_state:
        with st.spinner('Đang khởi tạo mô hình...'):
            st.session_state.model = DigitRecognizer()

    # Training section
    st.sidebar.header("Training Options")
    if st.sidebar.button("Train Model"):
        with st.spinner('Đang training mô hình...'):
            history = st.session_state.model.train()
            st.success("Training completed!")

            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['accuracy'], label='Train')
            ax1.plot(history.history['val_accuracy'], label='Validation')
            ax1.set_title('Model Accuracy')
            ax1.legend()

            ax2.plot(history.history['loss'], label='Train')
            ax2.plot(history.history['val_loss'], label='Validation')
            ax2.set_title('Model Loss')
            ax2.legend()

            st.pyplot(fig)

    # Upload and prediction section
    st.subheader("Upload ảnh chữ số")
    uploaded_file = st.file_uploader(
        "Chọn ảnh chứa chữ số viết tay", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã upload', width=200)

        # Nhận dạng
        if st.button("Nhận dạng"):
            digit, confidence, predictions = st.session_state.model.predict(
                image)

            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### Kết quả: {digit}")
                st.markdown(f"### Độ tin cậy: {confidence*100:.2f}%")

            with col2:
                # Vẽ biểu đồ confidence scores
                fig = plot_predictions(predictions)
                st.pyplot(fig)


if __name__ == "__main__":
    main()
