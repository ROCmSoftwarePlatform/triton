name: Triton-inductor unit tests

on:
  workflow_dispatch:
  pull_request:
    branches:
      - main
      - triton-mlir

jobs:
  Triton-Inductor-Tests:
    runs-on: ubuntu-latest
    
    container:
      image: rocm/pytorch-nightly:latest
      options: --user root --network=host --device=/dev/kfd --device=/dev/dri --ipc="host" --pid="host" --shm-size 8G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --user root
    env:
      TEST_SUITES: "inductor/test_standalone_compile inductor/test_torchinductor inductor/test_torchinductor_codegen_dynamic_shapes inductor/test_torchinductor_dynamic_shapes inductor/test_torchinductor_opinfo"

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      - name: Uninstall Triton Packages
        run: |
          pip3 uninstall -y triton
          pip3 uninstall -y pytorch-triton-rocm

      - name: Clear cache
        run: |
          rm -rf ~/.triton

      - name: Update PATH
        run: |
          echo "PATH=${HOME}/.local/bin:${PATH}" >> "${GITHUB_ENV}"

      - name: Install Triton
        run: |
          cd python
          pip3 install ninja
          # Install in system, because need to override system triton. Otherwise lit tests will use wrong version
          DEBUG=TRUE TRITON_USE_ROCM=TRUE TRITON_USE_ASSERT_ENABLED_LLVM=TRUE python3 -m pip install --no-build-isolation -vvv -e .

      - name: PyTorch UTs
        run: |
          cd /root/
          git clone https://github.com/pytorch/pytorch --recursive
          PYTORCH_TEST_WITH_ROCM=1 python3 pytorch/test/run_test.py --continue-through-error --verbose --include ${TEST_SUITES} 2>&1 | tee test_pytorch_inductor.log
