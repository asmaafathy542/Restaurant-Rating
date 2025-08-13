from tensorflow.keras.models import load_model

# اسم ملف الموديل القديم
old_model_path = "neural_model.h5"  # عدّل الاسم حسب ملفك

# تحميل الموديل بصيغته القديمة بدون compile
model = load_model(old_model_path, compile=False)

# حفظ الموديل بصيغة Keras الجديدة (.keras)
model.save("neural_model_fixed.keras")

print("تم حفظ الموديل الجديد: neural_model_fixed.keras")
