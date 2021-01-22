export PATH=$HOME/miniconda/bin/:$PATH
conda activate optima

echo "=== Configuring Optima..."
cmake -S . -B build -G Ninja
echo "=== Configuring Optima...finished!"

echo "=== Building and installing Optima..."
cmake --build build --target install
echo "=== Building and installing Optima...finished!"
