export PATH=$HOME/miniconda/bin/:$PATH
source activate optima

echo "=== Running tests..."
cmake --build build --target tests
echo "=== Running tests...finished!"
