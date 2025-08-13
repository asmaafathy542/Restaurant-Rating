from tensorflow.keras.models import load_model

# تحميل الموديل القديم
old_model = load_model("neural_model.h5", compile=False)

# حفظ بصيغة جديدة متوافقة مع Keras 3
old_model.save("neural_model_fixed.keras")
