#!/usr/bin/env bash

set -ex

# Check if ROCM_VERSION argument is provided
if [[ -z "$1" ]]; then
    echo "ROCM_VERSION argument not provided" && exit 1
else
    ROCM_VERSION="$1"
fi

# Set ROCM_HOME if not set
if [[ -z "${ROCM_HOME}" ]]; then
    export ROCM_HOME=/opt/rocm
fi

# Extract major and minor version numbers
MAJOR_VERSION=$(echo "${ROCM_VERSION}" | cut -d '.' -f 1)
MINOR_VERSION=$(echo "${ROCM_VERSION}" | cut -d '.' -f 2)
ROCM_INT=$(($MAJOR_VERSION * 10000 + $MINOR_VERSION * 100)) # Add patch later

# Check TRITON_ROCM_DIR is set
if [[ -z "${TRITON_ROCM_DIR}" ]]; then
  export TRITON_ROCM_DIR=python/triton/third_party/hip
fi

# Remove current libhsa included to avoid confusion
rm $TRITON_ROCM_DIR/lib/hsa/libhsa-runtime*

OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    LIBTINFO_PATH="/usr/lib64/libtinfo.so.5"
else
    LIBTINFO_PATH="/usr/lib64/libtinfo.so.6"
fi
LIBNUMA_PATH="/usr/lib64/libnuma.so.1"
LIBELF_PATH="/usr/lib64/libelf.so.1"

OS_SO_PATHS=(
    $LIBELF_PATH
    $LIBNUMA_PATH
    $LIBTINFO_PATH
)

for lib in "${OS_SO_PATHS[@]}"
do
    cp $lib $TRITON_ROCM_DIR/lib/
done


# Required ROCm libraries
if [[ "${MAJOR_VERSION}" == "6" ]]; then
    libamdhip="libamdhip64.so.6"
else
    libamdhip="libamdhip64.so.5"
fi

# Required ROCm libraries - ROCm 6.0
ROCM_SO=(
    "${libamdhip}"
    "libhsa-runtime64.so.1"
    "libamd_comgr.so.2"
    "libdrm.so.2"
    "libdrm_amdgpu.so.1"
)

if [[ $ROCM_INT -ge 60100 ]]; then
    ROCM_SO+=("librocprofiler-register.so.0")
fi	

# Find the SO libs dynamically
for lib in "${ROCM_SO[@]}"
do
    file_path=($(find $ROCM_HOME/lib/ -name "$lib")) # First search in lib
    if [[ -z $file_path ]]; then
        if [ -d "$ROCM_HOME/lib64/" ]; then
            file_path=($(find $ROCM_HOME/lib64/ -name "$lib")) # Then search in lib64
        fi
    fi
    if [[ -z $file_path ]]; then
        file_path=($(find $ROCM_HOME/ -name "$lib")) # Then search in ROCM_HOME
    fi
    if [[ -z $file_path ]]; then
        file_path=($(find /opt/ -name "$lib")) # Then search in ROCM_HOME
    fi
    if [[ -z $file_path ]]; then
            echo "Error: Library file $lib is not found." >&2
            exit 1
    fi

    cp $file_path $TRITON_ROCM_DIR/lib
    # When running locally, and not building a wheel, we need to satisfy shared objects requests that don't look for versions
    LINKNAME=$(echo $lib | sed -e 's/\.so.*/.so/g')
    ln -sf $lib $TRITON_ROCM_DIR/lib/$LINKNAME
done

# Copy Include Files
cp -r $ROCM_HOME/include/hip $TRITON_ROCM_DIR/include

# Copy linker
mkdir -p $TRITON_ROCM_DIR/llvm/bin
cp $ROCM_HOME/llvm/bin/ld.lld $TRITON_ROCM_DIR/llvm/bin/

