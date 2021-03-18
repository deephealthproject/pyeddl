# Recompile the EDDL
cd ~/Desktop/pyeddl_onnx_last/pyeddl/third_party/eddl/build
make -j 8
make install

# Erase the build folder to make sure the pyEDDL install is clean
cd ~/Desktop/pyeddl_onnx_last/pyeddl
rm -rf build

# Regenerate bindings
#sudo bash generate_bindings.sh

# Reinstall the pyEDDL
python3 setup.py install

# Back to the working folder
cd ~/Desktop/pyeddl_onnx_last/pyeddl/third_party/compss_runtime

# Execution of the python script
python3 eddl_train_batch_compss.py
