#!/bin/bash
set -e

# Usage:
#   ./build.sh breakout              # Build _C.so with breakout statically linked
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --cpu        # CPU fallback, torch only
#   ./build.sh breakout --mps        # Apple GPU training + CPU native simulator
#   ./build.sh breakout --debug      # Debug build
#   ./build.sh breakout --local      # Standalone executable (debug, sanitizers)
#   ./build.sh breakout --fast       # Standalone executable (optimized)
#   ./build.sh breakout --web        # Emscripten web build
#   ./build.sh breakout --profile    # Kernel profiling binary
#   ./build.sh matsci --mps --lammps # Explicit optional LAMMPS backend
#   ./build.sh all                   # Build every binding (CPU once, CUDA both precisions)

if [ -z "$1" ]; then
    echo "Usage: ./build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web|--profile|--cpu|--mps] [--lammps]"
    exit 1
fi
ENV=$1
shift

PLATFORM="$(uname -s)"
MACHINE="$(uname -m)"

for arg in "$@"; do
    case $arg in
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --debug) DEBUG=1 ;;
        --local) MODE=local ;;
        --fast)  MODE=fast ;;
        --web)   MODE=web ;;
        --profile) MODE=profile ;;
        --cpu)   MODE=cpu; PRECISION="-DPRECISION_FLOAT"; BUILD_MPS_BRIDGE=; DISABLE_MPS_BRIDGE=1 ;;
        # MPS executes PyTorch policy training on the Apple GPU while the
        # native simulator extension remains the portable CPU backend.
        --mps)   MODE=cpu; PRECISION="-DPRECISION_FLOAT"; BUILD_MPS_BRIDGE=1; DISABLE_MPS_BRIDGE= ;;
        # Matsci defaults to its equivalent native ballistic implementation.
        # LAMMPS is retained only as an explicitly linked reference backend.
        --lammps) USE_LAMMPS=1 ;;
        *) echo "Error: unknown argument '$arg'" && exit 1 ;;
    esac
done

# Apple Silicon has no CUDA toolchain. Make the useful native extension the
# default there; Linux and Intel macOS retain their existing mode selection.
if [ "$PLATFORM" = "Darwin" ] \
        && { [ "$MACHINE" = "arm64" ] || [ "$MACHINE" = "aarch64" ]; } \
        && [ -z "$MODE" ]; then
    MODE=cpu
    PRECISION="-DPRECISION_FLOAT"
    BUILD_MPS_BRIDGE=1
fi

if [ "$ENV" = "all" ]; then
    CHILD_ARGS=()
    [ -n "$DEBUG" ] && CHILD_ARGS+=(--debug)
    case "$MODE" in
        local) CHILD_ARGS+=(--local) ;;
        fast) CHILD_ARGS+=(--fast) ;;
        web) CHILD_ARGS+=(--web) ;;
        profile) CHILD_ARGS+=(--profile) ;;
        cpu)
            if [ -n "$BUILD_MPS_BRIDGE" ]; then
                CHILD_ARGS+=(--mps)
            else
                CHILD_ARGS+=(--cpu)
            fi
            ;;
    esac

    BUILD_FLOAT_VARIANT=1
    if [ "$MODE" = "cpu" ] || [ "$PRECISION" = "-DPRECISION_FLOAT" ]; then
        BUILD_FLOAT_VARIANT=0
        if [ "$MODE" != "cpu" ]; then
            CHILD_ARGS+=(--float)
        fi
    fi

    FAILED=()
    for env_dir in ocean/*/; do
        [ -f "${env_dir}binding.c" ] || continue
        env=$(basename "$env_dir")
        ENV_CHILD_ARGS=("${CHILD_ARGS[@]}")
        if [ -n "$USE_LAMMPS" ] && [ "$env" = "matsci" ]; then
            ENV_CHILD_ARGS+=(--lammps)
        fi
        env_failed=0
        bash "$0" "$env" "${ENV_CHILD_ARGS[@]}" || env_failed=1
        if [ "$BUILD_FLOAT_VARIANT" -eq 1 ]; then
            bash "$0" "$env" "${ENV_CHILD_ARGS[@]}" --float || env_failed=1
        fi

        if [ "$env_failed" -eq 0 ]; then
            echo "OK: $env"
        else
            echo "FAIL: $env"
            FAILED+=("$env")
        fi
    done

    if [ "${#FAILED[@]}" -gt 0 ]; then
        printf '\nFailed builds:\n'
        printf '  %s\n' "${FAILED[@]}"
        exit 1
    fi
    exit 0
fi

if [ -n "$USE_LAMMPS" ] && [ "$ENV" != "matsci" ]; then
    echo "Error: --lammps is valid only for the matsci environment"
    exit 1
fi

# Linux/mac
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP_COMPILE_FLAGS=(-fopenmp)
    OMP_LINK_FLAGS=(-fopenmp -lomp5)
    NVCC_OMP_FLAG=-Xcompiler=-fopenmp
    SANITIZE_FLAGS=(-fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer)
    STANDALONE_LDFLAGS=(-lGL)
    SHARED_LDFLAGS=(-Bsymbolic-functions)
else
    RAYLIB_NAME='raylib-5.5_macos'
    # Apple Clang does not discover Homebrew's libomp automatically. Prefer
    # Homebrew LLVM when available (it also knows the newest Apple CPUs), but
    # keep Apple Clang working via the explicit preprocessor and linker flags.
    LLVM_PREFIX="$(brew --prefix llvm 2>/dev/null || true)"
    LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
    if [ -z "${CC:-}" ] && [ -x "$LLVM_PREFIX/bin/clang" ]; then
        CC="$LLVM_PREFIX/bin/clang"
    fi
    if [ -z "${CXX:-}" ] && [ -x "$LLVM_PREFIX/bin/clang++" ]; then
        CXX="$LLVM_PREFIX/bin/clang++"
    fi
    if [ -z "$LIBOMP_PREFIX" ] || [ ! -f "$LIBOMP_PREFIX/include/omp.h" ]; then
        echo "Error: libomp is required on macOS (brew install libomp)"
        exit 1
    fi
    OMP_COMPILE_FLAGS=(-Xpreprocessor -fopenmp -I"$LIBOMP_PREFIX/include")
    OMP_LINK_FLAGS=(-L"$LIBOMP_PREFIX/lib" -Wl,-rpath,"$LIBOMP_PREFIX/lib" -lomp)
    PYTHON_OMP_LINK_FLAGS=("${OMP_LINK_FLAGS[@]}")
    TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
    TORCH_OMP_ID=
    if [ -f "$TORCH_LIB_DIR/libomp.dylib" ]; then
        # Reuse PyTorch's exact OpenMP runtime in the extension. Loading both
        # this and Homebrew's libomp aborts at runtime on macOS.
        PYTHON_OMP_LINK_FLAGS=(-L"$TORCH_LIB_DIR" -Wl,-rpath,"$TORCH_LIB_DIR" -lomp)
        TORCH_OMP_ID="$(otool -D "$TORCH_LIB_DIR/libomp.dylib" | tail -n 1 | xargs)"
    fi
    NVCC_OMP_FLAG=
    SANITIZE_FLAGS=()
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
    SHARED_LDFLAGS=(-framework Cocoa -framework OpenGL -framework IOKit -undefined dynamic_lookup)
fi

if [ "$PLATFORM" != "Darwin" ]; then
    PYTHON_OMP_LINK_FLAGS=("${OMP_LINK_FLAGS[@]}")
fi

VISIBILITY_FLAGS=(-fvisibility=hidden)
if [ "$PLATFORM" != "Darwin" ]; then
    VISIBILITY_FLAGS+=(-fno-semantic-interposition)
fi

fix_python_openmp_link() {
    local target=$1
    if [ "$PLATFORM" = "Darwin" ] && [ -n "${TORCH_OMP_ID:-}" ]; then
        install_name_tool -change "$TORCH_OMP_ID" @rpath/libomp.dylib "$target"
    fi
}

CLANG_WARN=(
    -Wall
    -ferror-limit=3
    -Werror=incompatible-pointer-types
    -Werror=return-type
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
)

download() {
    local name=$1 url=$2
    [ -d "$name" ] && return
    echo "Downloading $name..."
    case "$url" in
        *.zip) curl -sL "$url" -o "$name.zip" && unzip -q "$name.zip" && rm "$name.zip" ;;
        *)     curl -sL "$url" -o "$name.tar.gz" && tar xf "$name.tar.gz" && rm "$name.tar.gz" ;;
    esac
}

RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.zip"
else
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.tar.gz"
fi

RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"
INCLUDES=(-I./$RAYLIB_NAME/include -I./src -I./vendor)
LINK_ARCHIVES=("$RAYLIB_A")
EXTRA_SRC=""
EXTRA_LDFLAGS=()
ENV_CFLAGS=()

if [ "$ENV" = "constellation" ]; then
    SRC_DIR="constellation"
    EXTRA_SRC="vendor/cJSON.c"
    OUTPUT_NAME="seethestars"
elif [ "$ENV" = "trailer" ]; then
    SRC_DIR="trailer"
    OUTPUT_NAME="trailer/trailer"
elif [ "$ENV" = "impulse_wars" ]; then
    SRC_DIR="ocean/$ENV"
    if [ "$MODE" = "web" ]; then BOX2D_NAME='box2d-web'
    elif [ "$PLATFORM" = "Linux" ]; then BOX2D_NAME='box2d-linux-amd64'
    else BOX2D_NAME='box2d-macos-arm64'
    fi
    BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"
    download "$BOX2D_NAME" "$BOX2D_URL/$BOX2D_NAME.tar.gz"
    INCLUDES+=(-I./$BOX2D_NAME/include -I./$BOX2D_NAME/src)
    LINK_ARCHIVES+=("./$BOX2D_NAME/libbox2d.a")
elif [ "$ENV" = "nethack" ]; then
    SRC_DIR="ocean/$ENV"
    NLE_DIR="vendor/fast-nle"
    NLE_REPO="https://github.com/FinlaySanders/fast-nle.git"
    if [ ! -d "$NLE_DIR/src" ]; then
        echo "Cloning fast-nle from $NLE_REPO ..."
        git clone --depth 1 "$NLE_REPO" "$NLE_DIR"
    fi
    NETHACK_LIB_DIR="$(pwd)/$NLE_DIR/build"
    NETHACK_RUNTIME_DIR="$(pwd)/$NLE_DIR/runtime"
    NETHACK_CACHE="$NETHACK_LIB_DIR/CMakeCache.txt"
    if [ ! -f "$NETHACK_LIB_DIR/libnethack.so" ] \
            || ! grep -Fqx "HACKDIR:STRING=$NETHACK_RUNTIME_DIR" \
                "$NETHACK_CACHE" 2>/dev/null; then
        echo "Building libnethack.so ..."
        BUILD_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
        if ! [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
            BUILD_JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
        fi
        [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || BUILD_JOBS=1
        cmake -S "$NLE_DIR" -B "$NETHACK_LIB_DIR" \
            -DCMAKE_BUILD_TYPE=Release -DHACKDIR="$NETHACK_RUNTIME_DIR"
        cmake --build "$NETHACK_LIB_DIR" --target nethack \
            --parallel "$BUILD_JOBS"
    fi
    INCLUDES+=(-I./$NLE_DIR/include
               -I./$NLE_DIR/build/_deps/deboost_context-src/include)
    EXTRA_LDFLAGS+=(-L"$NETHACK_LIB_DIR" -lnethack -Wl,-rpath,"$NETHACK_LIB_DIR" -ldl)
elif [ "$ENV" = "matsci" ]; then
    SRC_DIR="ocean/$ENV"
    if [ -n "$USE_LAMMPS" ]; then
        if [ "$MODE" = "web" ]; then
            echo "Error: the Matsci LAMMPS backend is unavailable for WebAssembly"
            exit 1
        fi
        ENV_CFLAGS+=(-DPUFFERLIB_USE_LAMMPS=1)

        LAMMPS_PKG=""
        if [ -z "${LAMMPS_INCLUDE_DIR:-}" ] \
                && [ -z "${LAMMPS_LIB_DIR:-}" ] \
                && [ -z "${LAMMPS_ROOT:-}" ] \
                && command -v pkg-config >/dev/null; then
            for candidate in lammps liblammps; do
                if pkg-config --exists "$candidate"; then
                    LAMMPS_PKG="$candidate"
                    break
                fi
            done
        fi
        if [ -n "$LAMMPS_PKG" ]; then
            read -r -a LAMMPS_CFLAGS_ARRAY \
                <<< "$(pkg-config --cflags "$LAMMPS_PKG")"
            read -r -a LAMMPS_LIBS_ARRAY \
                <<< "$(pkg-config --libs "$LAMMPS_PKG")"
            INCLUDES+=("${LAMMPS_CFLAGS_ARRAY[@]}")
            EXTRA_LDFLAGS+=("${LAMMPS_LIBS_ARRAY[@]}")
            for flag in "${LAMMPS_LIBS_ARRAY[@]}"; do
                if [[ "$flag" == -L* ]]; then
                    EXTRA_LDFLAGS+=(-Wl,-rpath,"${flag#-L}")
                fi
            done
        else
            LAMMPS_INCLUDE="${LAMMPS_INCLUDE_DIR:-}"
            if [ -z "$LAMMPS_INCLUDE" ] && [ -n "${LAMMPS_ROOT:-}" ]; then
                LAMMPS_INCLUDE="$LAMMPS_ROOT/include"
            fi
            if [ -z "$LAMMPS_INCLUDE" ]; then
                for candidate in /opt/homebrew/include /usr/local/include /usr/include; do
                    if [ -f "$candidate/lammps/library.h" ]; then
                        LAMMPS_INCLUDE="$candidate"
                        break
                    fi
                done
            fi

            LAMMPS_LIBRARY="${LAMMPS_LIB_DIR:-}"
            if [ -z "$LAMMPS_LIBRARY" ] && [ -n "${LAMMPS_ROOT:-}" ]; then
                LAMMPS_LIBRARY="$LAMMPS_ROOT/lib"
            fi
            if [ -z "$LAMMPS_LIBRARY" ]; then
                for candidate in \
                        /opt/homebrew/lib /usr/local/lib /usr/lib \
                        /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu; do
                    if [ -f "$candidate/liblammps.dylib" ] \
                            || [ -f "$candidate/liblammps.so" ] \
                            || [ -f "$candidate/liblammps.a" ]; then
                        LAMMPS_LIBRARY="$candidate"
                        break
                    fi
                done
            fi

            if [ -z "$LAMMPS_INCLUDE" ] || [ -z "$LAMMPS_LIBRARY" ] \
                    || [ ! -f "$LAMMPS_INCLUDE/lammps/library.h" ] \
                    || { [ ! -f "$LAMMPS_LIBRARY/liblammps.dylib" ] \
                        && [ ! -f "$LAMMPS_LIBRARY/liblammps.so" ] \
                        && [ ! -f "$LAMMPS_LIBRARY/liblammps.a" ]; }; then
                echo "Error: --lammps requested but headers/libraries were not found."
                echo "Install a LAMMPS library with lammps/liblammps pkg-config metadata, or set LAMMPS_INCLUDE_DIR and LAMMPS_LIB_DIR."
                exit 1
            fi
            INCLUDES+=(-I"$LAMMPS_INCLUDE")
            EXTRA_LDFLAGS+=(-L"$LAMMPS_LIBRARY" -Wl,-rpath,"$LAMMPS_LIBRARY" -llammps)
        fi
        echo "Matsci backend: LAMMPS (explicit --lammps)"
    else
        echo "Matsci backend: native ballistic (periodic box, pair_style zero equivalent)"
    fi
elif [ -d "ocean/$ENV" ]; then
    SRC_DIR="ocean/$ENV"
else
    echo "Error: environment '$ENV' not found" && exit 1
fi

OUTPUT_NAME=${OUTPUT_NAME:-$ENV}

# Standalone environment build
# Native builds are consumed on the machine that compiled them. x86_64 keeps
# the historical AVX2/FMA floor; ARM64 enables the local Apple/ARM CPU features
# (NEON, dot-product, BF16, etc.) without passing invalid x86 flags.
if [ "$MACHINE" = "x86_64" ] || [ "$MACHINE" = "amd64" ]; then
    SIMD_FLAGS=(-mavx2 -mfma)
elif [ "$MACHINE" = "arm64" ] || [ "$MACHINE" = "aarch64" ]; then
    SIMD_FLAGS=(-mcpu=native)
    if [ "$PLATFORM" = "Darwin" ] && [[ "$(sysctl -n machdep.cpu.brand_string 2>/dev/null)" == *"M5"* ]] \
            && "${CC:-clang}" --print-supported-cpus 2>/dev/null | grep -q 'apple-m5'; then
        SIMD_FLAGS=(-mcpu=apple-m5)
    fi
else
    SIMD_FLAGS=()
fi
if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT=(-g -O0 "${CLANG_WARN[@]}" "${SANITIZE_FLAGS[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O0 -g"
    LINK_OPT="-g"
else
    CLANG_OPT=(-O2 -DNDEBUG "${CLANG_WARN[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O2 --threads 0"
    LINK_OPT="-O2"
fi
if [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
    FLAGS=(
        "${INCLUDES[@]}"
        "${ENV_CFLAGS[@]}"
        "$SRC_DIR/$ENV.c" $EXTRA_SRC -o "$OUTPUT_NAME"
        "${LINK_ARCHIVES[@]}"
        "${EXTRA_LDFLAGS[@]}"
        "${STANDALONE_LDFLAGS[@]}"
        -lm -lpthread
        "${OMP_COMPILE_FLAGS[@]}"
        "${OMP_LINK_FLAGS[@]}"
        -DPLATFORM_DESKTOP
    )
    echo "Compiling $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" "${FLAGS[@]}"
    echo "Built: ./$OUTPUT_NAME"
    exit 0
elif [ "$MODE" = "web" ]; then
    mkdir -p "build/web/$ENV"
    echo "Compiling $ENV for web..."
    emcc \
        -o "build/web/$ENV/game.html" \
        "$SRC_DIR/$ENV.c" $EXTRA_SRC \
        -O3 -Wall \
        "${LINK_ARCHIVES[@]}" \
        "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --shell-file vendor/minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DNDEBUG -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file resources/$ENV@resources/$ENV \
        --preload-file resources/shared@resources/shared
    echo "Built: build/web/$ENV/game.html"
    exit 0
fi

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_COMPILERCHECK=content
CC="${CC:-$(command -v ccache >/dev/null && echo 'ccache clang' || echo 'clang')}"

# CPU/MPS builds do not need a CUDA installation or any of the optional
# NVIDIA Python wheels. Keep their discovery out of this path entirely.
CUDA_INCLUDE_FLAGS=()
if [ "$MODE" != "cpu" ]; then
    CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
    if [ -z "$CUDA_HOME" ]; then
        NVCC_PATH="$(command -v nvcc || true)"
        if [ -z "$NVCC_PATH" ]; then
            echo "Error: nvcc not found; use --cpu (or --mps on Apple Silicon)"
            exit 1
        fi
        CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
    fi
    CUDA_INCLUDE_FLAGS=(-I"$CUDA_HOME/include")

    # Find cuDNN path.
    CUDNN_IFLAG=""
    CUDNN_LFLAG=""
    for dir in /usr/local/cuda/include /usr/include; do
        if [ -f "$dir/cudnn.h" ]; then
            CUDNN_IFLAG="-I$dir"
            break
        fi
    done
    for dir in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
        if [ -f "$dir/libcudnn.so" ]; then
            CUDNN_LFLAG="-L$dir"
            break
        fi
    done
    if [ -z "$CUDNN_IFLAG" ]; then
        CUDNN_IFLAG=$(python -c "import nvidia.cudnn, os; print('-I' + os.path.join(nvidia.cudnn.__path__[0], 'include'))" 2>/dev/null || echo "")
    fi
    if [ -z "$CUDNN_LFLAG" ]; then
        CUDNN_LFLAG=$(python -c "import nvidia.cudnn, os; print('-L' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null || echo "")
    fi

    # NCCL include/lib fallback (mirrors the cuDNN fallback above).
    # Needed when NCCL is provided by the nvidia-nccl-cu12 wheel in the active venv.
    NCCL_IFLAG=""
    NCCL_LFLAG=""
    for dir in /usr/include /usr/local/cuda/include; do
        if [ -f "$dir/nccl.h" ]; then NCCL_IFLAG="-I$dir"; break; fi
    done
    for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
        if [ -f "$dir/libnccl.so" ] || [ -f "$dir/libnccl.so.2" ]; then NCCL_LFLAG="-L$dir"; break; fi
    done
    if [ -z "$NCCL_IFLAG" ]; then
        NCCL_IFLAG=$(python -c "import nvidia.nccl, os; print('-I' + os.path.join(nvidia.nccl.__path__[0], 'include'))" 2>/dev/null || echo "")
    fi
    if [ -z "$NCCL_LFLAG" ]; then
        NCCL_LFLAG=$(python -c "import nvidia.nccl, os; print('-L' + os.path.join(nvidia.nccl.__path__[0], 'lib'))" 2>/dev/null || echo "")
    fi

    WHEEL_RPATH_FLAGS=()
    for lib_flag in "$CUDNN_LFLAG" "$NCCL_LFLAG"; do
        if [[ "$lib_flag" == -L* ]]; then
            WHEEL_RPATH_FLAGS+=("-Wl,-rpath,${lib_flag#-L}")
        fi
    done

    if command -v ccache >/dev/null; then
        NVCC="ccache $CUDA_HOME/bin/nvcc"
    else
        NVCC="$CUDA_HOME/bin/nvcc"
    fi
    ARCH=${NVCC_ARCH:-native}
    NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
fi

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

build_mps_rng_bridge() {
    local bridge_output="pufferlib/_mps_rng${EXT_SUFFIX}"
    local bridge_candidate="build/_mps_rng${EXT_SUFFIX}"
    local bridge_stale="build/_mps_rng_stale${EXT_SUFFIX}"
    if [ -n "$DISABLE_MPS_BRIDGE" ] \
            && [ "$PLATFORM" = "Darwin" ] \
            && { [ "$MACHINE" = "arm64" ] || [ "$MACHINE" = "aarch64" ]; }; then
        if [ -f "$bridge_output" ]; then
            mv -f "$bridge_output" "$bridge_stale"
            echo "Disabled stale MPS RNG bridge for explicit --cpu build."
        fi
        return
    fi
    if [ -z "$BUILD_MPS_BRIDGE" ] \
            || [ "$PLATFORM" != "Darwin" ] \
            || { [ "$MACHINE" != "arm64" ] && [ "$MACHINE" != "aarch64" ]; }; then
        return
    fi

    if ! python -c \
            'import torch; assert str(torch.__version__).split("+", 1)[0] == "2.13.0"; assert torch.version.git_version == "cf30153c4c131c8164ee7798e5022d810682e2cb"' \
            >/dev/null 2>&1; then
        echo "Warning: fused MPS sampling requires the validated PyTorch 2.13.0 build; using torch.multinomial fallback."
        if [ -f "$bridge_output" ]; then
            mv -f "$bridge_output" "$bridge_stale"
        fi
        return
    fi
    if [ -f "$bridge_output" ] && [ "$bridge_output" -nt src/mps_rng.cpp ] \
            && python -c \
                'import torch; from pufferlib import _mps_rng; assert callable(_mps_rng.reserve_default_mps_philox)' \
                >/dev/null 2>&1; then
        echo "MPS RNG reservation bridge is current: $bridge_output"
        return
    fi
    local torch_include_flags=()
    local torch_library_flags=()
    local path
    while IFS= read -r path; do
        [ -n "$path" ] && torch_include_flags+=("-I$path")
    done < <(python -c \
        'from torch.utils.cpp_extension import include_paths; print("\n".join(include_paths()))')
    while IFS= read -r path; do
        if [ -n "$path" ]; then
            torch_library_flags+=("-L$path" "-Wl,-rpath,$path")
        fi
    done < <(python -c \
        'from torch.utils.cpp_extension import library_paths; print("\n".join(library_paths()))')

    echo "Compiling atomic MPS RNG reservation bridge..."
    if ! ${CXX:-clang++} -std=c++17 -fPIC -shared \
        -undefined dynamic_lookup \
        "${torch_include_flags[@]}" \
        -I$PYTHON_INCLUDE \
        src/mps_rng.cpp \
        "${torch_library_flags[@]}" \
        -ltorch -ltorch_cpu -ltorch_python -lc10 \
        $LINK_OPT \
        -o "$bridge_candidate"; then
        echo "Warning: atomic MPS RNG bridge compilation failed; using torch.multinomial fallback."
        if [ -f "$bridge_output" ]; then
            mv -f "$bridge_output" "$bridge_stale"
        fi
        return
    fi
    if ! PUFFER_MPS_BRIDGE_CANDIDATE="$bridge_candidate" python -c \
        'import importlib.util, os, torch; p = os.environ["PUFFER_MPS_BRIDGE_CANDIDATE"]; s = importlib.util.spec_from_file_location("_mps_rng", p); assert s is not None and s.loader is not None; m = importlib.util.module_from_spec(s); s.loader.exec_module(m); assert callable(m.reserve_default_mps_philox)'; then
        echo "Warning: atomic MPS RNG bridge smoke import failed; using torch.multinomial fallback."
        if [ -f "$bridge_output" ]; then
            mv -f "$bridge_output" "$bridge_stale"
        fi
        return
    fi
    mv -f "$bridge_candidate" "$bridge_output"
    echo "Built: $bridge_output"
}

BINDING_SRC="$SRC_DIR/binding.c"
mkdir -p build
STATIC_OBJ="build/libstatic_${ENV}.o"
STATIC_LIB="build/libstatic_${ENV}.a"

if [ ! -f "$BINDING_SRC" ]; then
    echo "Error: $BINDING_SRC not found"
    exit 1
fi

echo "Compiling static library for $ENV..."
${CC:-clang} -c "${CLANG_OPT[@]}" $EXTRA_CFLAGS \
    -Isrc -I$SRC_DIR -Ivendor \
    "${INCLUDES[@]}" \
    -I./$RAYLIB_NAME/include "${CUDA_INCLUDE_FLAGS[@]}" \
    -DPLATFORM_DESKTOP \
    "${ENV_CFLAGS[@]}" \
    "${VISIBILITY_FLAGS[@]}" \
    -fPIC "${OMP_COMPILE_FLAGS[@]}" \
    "$BINDING_SRC" -o "$STATIC_OBJ"
ar rcs "$STATIC_LIB" "$STATIC_OBJ"

# Brittle hack: have to extract the tensor type from the static lib to build trainer
OBS_TENSOR_T=$(awk '/^#define OBS_TENSOR_T/{print $3}' "$BINDING_SRC")
if [ -z "$OBS_TENSOR_T" ]; then
    echo "Error: Could not find OBS_TENSOR_T in $BINDING_SRC"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "Compiling CUDA ($ARCH) training backend..."
    $NVCC -c -arch=$ARCH -Xcompiler -fPIC \
        -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1 \
        -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
        -Xcompiler=-DPLATFORM_DESKTOP \
        -std=c++17 \
        -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE -I$NUMPY_INCLUDE \
        -I$CUDA_HOME/include $CUDNN_IFLAG $NCCL_IFLAG -I$RAYLIB_NAME/include \
        $NVCC_OMP_FLAG \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        $PRECISION $NVCC_OPT \
        src/bindings.cu -o build/bindings.o

    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC
        build/bindings.o "$STATIC_LIB" "${LINK_ARCHIVES[@]}"
        -L$CUDA_HOME/lib64 $CUDNN_LFLAG $NCCL_LFLAG
        "${WHEEL_RPATH_FLAGS[@]}"
        "${EXTRA_LDFLAGS[@]}"
        -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
        "${PYTHON_OMP_LINK_FLAGS[@]}" $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    fix_python_openmp_link "$OUTPUT"
    PUFFER_BUILD_ENV="$ENV" python -c \
        'import os; from pufferlib import _C; assert _C.env_name == os.environ["PUFFER_BUILD_ENV"]'
    build_mps_rng_bridge
    echo "Built: $OUTPUT"

elif [ "$MODE" = "cpu" ]; then
    echo "Compiling CPU training backend..."
    ${CXX:-g++} -c -fPIC "${OMP_COMPILE_FLAGS[@]}" \
        -D_GLIBCXX_USE_CXX11_ABI=1 \
        -DPLATFORM_DESKTOP \
        -std=c++17 \
        -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        $PRECISION $LINK_OPT \
        src/bindings_cpu.cpp -o build/bindings_cpu.o
    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC
        build/bindings_cpu.o "$STATIC_LIB" "${LINK_ARCHIVES[@]}"
        "${EXTRA_LDFLAGS[@]}"
        -lm -lpthread "${PYTHON_OMP_LINK_FLAGS[@]}" $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    fix_python_openmp_link "$OUTPUT"
    PUFFER_BUILD_ENV="$ENV" python -c \
        'import os; from pufferlib import _C; assert _C.env_name == os.environ["PUFFER_BUILD_ENV"]'
    build_mps_rng_bridge
    echo "Built: $OUTPUT"

elif [ "$MODE" = "profile" ]; then
    echo "Compiling profile binary ($ARCH)..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -Isrc -I$SRC_DIR -Ivendor \
        -I$CUDA_HOME/include $CUDNN_IFLAG $NCCL_IFLAG -I$RAYLIB_NAME/include \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        -Xcompiler=-DPLATFORM_DESKTOP \
        $PRECISION \
        $NVCC_OMP_FLAG \
        tests/profile_kernels.cu vendor/ini.c \
        "$STATIC_LIB" "${LINK_ARCHIVES[@]}" \
        -lnccl -lnvidia-ml -lcublas -lcurand -lcudnn \
        -lGL -lm -lpthread "${OMP_LINK_FLAGS[@]}" \
        -o profile
    echo "Built: ./profile"
fi
