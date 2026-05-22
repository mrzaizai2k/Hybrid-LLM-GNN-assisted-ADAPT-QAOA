# **ADAPT-GPT Full Setup Guide (WSL2 / Linux)**

## **1️⃣ Install System Dependencies**

Open your terminal and run:

```bash
# Update package list
sudo apt update

# Install build essentials and required system libraries
sudo apt install -y build-essential curl wget git libssl-dev libcurl4-openssl-dev \
libgit2-dev zlib1g-dev pkg-config
```

### **Check installed versions**

```bash
git --version
curl --version
openssl version
```

Make sure all commands return versions correctly.

---

## **2️⃣ Install Julia**

1. Download the official Julia tarball (1.12.1 recommended):

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.1-linux-x86_64.tar.gz
```

2. Extract and move to `/opt`:

```bash
tar -xvzf julia-1.12.1-linux-x86_64.tar.gz
sudo mv julia-1.12.1 /opt/
```

3. Add Julia to PATH:

```bash
echo 'export PATH="/opt/julia-1.12.1/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

4. Verify installation:

```bash
julia --version
```

---

## **3️⃣ Clean old Julia caches (optional but recommended)**

Remove old compiled packages and broken package directories:

```bash
rm -rf ~/.julia/compiled/v1.12/*
rm -rf ~/.julia/packages/LibSSH2_jll*
rm -rf ~/.julia/packages/LibGit2_jll*
rm -rf ~/.julia/packages/LibCURL_jll*
```

---

## **4️⃣ Setup Python Environment**

1. Install **Miniconda / Anaconda** if not already installed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

2. Create a Python 3.10 environment for ADAPT-GPT:

```bash
conda create -n adapt_gpt python=3.10 -y
conda activate adapt_gpt
```

3. Install Python dependencies:

```bash
pip install torch numpy transformers datasets tiktoken wandb ipykernel pandas tqdm networkx matplotlib joblib scipy gurobipy
```

---

## **5️⃣ Clone ADAPT-GPT Repository**

```bash
git clone https://github.com/IlyaTyagin/ADAPT-GPT --recurse-submodules
cd ADAPT-GPT/ADAPT.jl/
```

---

## **6️⃣ Setup Julia Project**

1. Start Julia in project mode **without precompilation**:

```bash
julia --startup-file=no --history-file=no --project=.
```

2. Inside Julia REPL, import `Pkg` **without triggering precompilation**:

```julia
import Pkg
```

3. Instantiate all packages:

```julia
Pkg.instantiate()
```

4. Rebuild low-level system libraries:

```julia
Pkg.build("OpenSSL_jll")
Pkg.build("LibSSH2_jll")
Pkg.build("LibGit2_jll")
Pkg.build("LibCURL_jll")
Pkg.precompile()
```

5. Add remaining ADAPT-GPT dependencies:

```julia
Pkg.add(["JuMP", "MQLib", "ProgressBars", "SimpleWeightedGraphs", "CSV", "DataFrames", "JSON", "ArgParse", "Multibreak"])
Pkg.develop(path="SciPyOptimizers")
Pkg.instantiate()
Pkg.precompile()
```

---

## **7️⃣ Verify Installation**

* Test Python packages:

```bash
python -c "import torch; import transformers; print('Python packages OK')"
```

* Test Julia packages:

```julia
using Pkg
Pkg.status()
println("Julia packages OK")
```

* Ensure no precompilation errors for `LibSSH2_jll`, `LibGit2_jll`, `LibCURL_jll`.

---

## **8️⃣ Optional: Use Julia Kernel in Jupyter**

```julia
import Pkg
Pkg.add("IJulia")
```

Then you can open Jupyter and select Julia as a kernel.

---

### ✅ **Notes / Tips**

* Always run Julia in `--project=.` mode to avoid global conflicts.
* If you ever encounter precompilation errors again, **delete the compiled caches** (`~/.julia/compiled/v1.12/*`) before rebuilding.
* For WSL2 users, make sure your system libraries (`libssl-dev`, `libgit2-dev`, `libcurl4-openssl-dev`) are up-to-date.
