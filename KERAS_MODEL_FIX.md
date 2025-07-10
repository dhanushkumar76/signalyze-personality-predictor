# 🔧 Keras Model Loading Fix

## 📋 **Issue Summary**
Your Streamlit app was experiencing a recurring error: `"Could not locate function 'weighted_loss_trait_1'"`. This happened because custom loss functions weren't properly registered for Keras serialization.

## ✅ **What Has Been Fixed**

### 1. **Created Custom Objects Module** (`utils/custom_objects.py`)
- ✅ All 8 trait-specific loss functions properly defined
- ✅ Functions registered with `@keras.saving.register_keras_serializable()`
- ✅ Focal loss used for traits 1, 3, 6, 7 (difficult traits)
- ✅ Standard CategoricalCrossentropy for traits 2, 4, 5, 8

### 2. **Updated Streamlit App** (`streamlit_app.py`)
- ✅ Added import: `from utils.custom_objects import CUSTOM_OBJECTS`
- ✅ Updated model loading: `load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)`

### 3. **Updated Training Script** (`scripts/3_train_model.py`)
- ✅ Imports registered loss functions
- ✅ Uses proper loss mapping
- ✅ Simplified callback system

## 🚀 **Your Options Now**

### **Option 1: Test Current Model (Try This First)**
The fix might work with your existing model:

```bash
# Activate your virtual environment first
source venv/Scripts/activate  # Windows
# OR
source venv/bin/activate      # Linux/Mac

# Run the app
streamlit run streamlit_app.py
```

**If this works:** 🎉 You're all set! The issue is resolved.

**If this doesn't work:** You'll need Option 2.

### **Option 2: Retrain the Model (If Option 1 Fails)**
If your current model still has serialization issues, retrain it:

```bash
# Make sure you're in your virtual environment
source venv/Scripts/activate

# Run the updated training script
python scripts/3_train_model.py
```

This will create a new model file that's properly serialized with the registered custom objects.

## 🔍 **How to Verify the Fix**

### Test Script Available
I've created `test_model_loading.py` to verify the fix:

```bash
python test_model_loading.py
```

This will:
- ✅ Load the model with custom objects
- ✅ Display model summary
- ✅ Confirm successful loading

## 📊 **Technical Details**

### **Before (Problem):**
```python
# Dynamic loss functions created in training script
def make_weighted_loss(key, base_loss):
    def loss(y_true, y_pred): 
        return trait_weights[key] * base_loss(y_true, y_pred)
    return loss

# Not registered for serialization ❌
loss_map = {f"trait_{i+1}": make_weighted_loss(...)}
```

### **After (Solution):**
```python
# Properly registered functions
@keras.saving.register_keras_serializable(package="custom_losses")
def weighted_loss_trait_1(y_true, y_pred):
    return focal_loss(y_true, y_pred)

# Properly loaded ✅
model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
```

## 🔮 **Future Prevention**

To avoid this issue in the future:
1. **Always register custom functions** with `@keras.saving.register_keras_serializable()`
2. **Use the custom_objects module** for all model loading
3. **Test model loading** after training with the test script

## 🆘 **If You Still Have Issues**

If you continue to experience problems:

1. **Check Python Environment:**
   ```bash
   which python
   pip list | grep tensorflow
   ```

2. **Reinstall Dependencies:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. **Clear Keras Cache:**
   ```bash
   rm -rf ~/.keras/models/*
   ```

4. **Verify File Paths:**
   - Ensure `model/best_model.keras` exists
   - Check that `utils/custom_objects.py` is properly imported

## 📝 **Summary**

The recurring Keras model loading error has been fixed by:
- ✅ Properly registering custom loss functions
- ✅ Creating a reusable custom objects module
- ✅ Updating both training and inference code
- ✅ Providing a test script for verification

Your Streamlit app should now load without the `"weighted_loss_trait_1"` error!