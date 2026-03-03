# Get Start
1. `/usr/bin/sqlite3 product.db < product.sql`
2. `bash product.sh`

# Create a view using the cli script `create_view.py`.
- `python create_view.py onnx <onnx_file> --layers=layers.json --profile=profile.json`
- `python create_view.py onnx <onnx_file> --layers=layers.json --profile=profile.json --layers2=layers2.json --profile2=profile2.json`
- `python create_view.py trex <layers_file> --profile=profile.json`

# Debugging
1. `bash debug.sh`